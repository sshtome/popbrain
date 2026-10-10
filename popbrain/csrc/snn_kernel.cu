#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel 1: Neuron update and sparse matvec (NO trace update here!)
__global__ void snn_update_neurons_kernel(
    const int N,
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col_idx,
    const float* __restrict__ W,
    const float* __restrict__ spikes_in,
    float* __restrict__ V,
    float* __restrict__ a,
    float* __restrict__ refrac_until,
    float* __restrict__ fired_out,
    const float t,
    const float Vrest,
    const float Vth,
    const float Vreset,
    const float refrac,
    const float decay,
    const float a_inc
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float v_add = 0.0f;
    int64_t start = crow[i];
    int64_t end = crow[i+1];
    for (int64_t e = start; e < end; ++e) {
        int64_t j = col_idx[e];
        v_add += W[e] * spikes_in[j];
    }

    float v = V[i];
    v = Vrest + (v + v_add - Vrest) * decay;

    bool fired = (v - a[i] >= Vth) && (t >= refrac_until[i]);
    float fired_f = fired ? 1.0f : 0.0f;

    if (fired) {
        v = Vreset;
        refrac_until[i] = t + refrac;
        a[i] += a_inc;
    }

    V[i] = v;
    fired_out[i] = fired_f;
}

// Kernel 2: Continuous STDP weight update using OLD traces
__global__ void snn_apply_stdp_kernel(
    const int N,
    const int64_t* __restrict__ crow,
    const int64_t* __restrict__ col_idx,
    float* __restrict__ W,
    const float* __restrict__ plastic,
    const int64_t* __restrict__ fly_of_syn,
    const float* __restrict__ reward,
    const float* __restrict__ fired,
    const float* __restrict__ pre_trace,
    const float* __restrict__ post_trace,
    const float eta_pot,
    const float eta_dep,
    const float dan_gain
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // i is the POST-synaptic neuron (row in CSR)
    float post_fired = fired[i];
    float post_tr = post_trace[i];

    int64_t start = crow[i];
    int64_t end = crow[i+1];
    
    for (int64_t e = start; e < end; ++e) {
        int64_t j = col_idx[e];
        // j is the PRE-synaptic neuron (column in CSR)
        float pre_fired = fired[j];
        
        if (pre_fired == 0.0f && post_fired == 0.0f) continue;
        
        float pre_tr = pre_trace[j];
        
        // Biological STDP:
        // If post fires, we potentiate (LTP) proportional to pre trace.
        // If pre fires, we depress (LTD) proportional to post trace.
        float dw = post_fired * (pre_tr * eta_pot) - pre_fired * (post_tr * eta_dep);
        
        if (dw != 0.0f) {
            float mod = 1.0f + dan_gain * reward[fly_of_syn[e]] * plastic[e];
            W[e] += dw * mod;
        }
    }
}

// Kernel 3: Trace update (runs AFTER STDP)
__global__ void snn_update_traces_kernel(
    const int N,
    float* __restrict__ pre_trace,
    float* __restrict__ post_trace,
    const float* __restrict__ fired_out,
    const float pre_decay,
    const float post_decay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float fired_f = fired_out[i];
    pre_trace[i] = (pre_trace[i] + fired_f) * pre_decay;
    post_trace[i] = (post_trace[i] + fired_f) * post_decay;
}

void forward_step(
    torch::Tensor crow, torch::Tensor col_idx, torch::Tensor W, torch::Tensor plastic,
    torch::Tensor fly_of_syn, torch::Tensor spikes_in, torch::Tensor V, torch::Tensor a,
    torch::Tensor refrac_until, torch::Tensor pre_trace, torch::Tensor post_trace,
    torch::Tensor reward, torch::Tensor fired_out,
    float t, float Vrest, float Vth, float Vreset, float refrac,
    float decay, float a_inc, float pre_decay, float post_decay,
    float eta_pot, float eta_dep, float dan_gain
) {
    const int N = V.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    snn_update_neurons_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        spikes_in.data_ptr<float>(), V.data_ptr<float>(), a.data_ptr<float>(),
        refrac_until.data_ptr<float>(), fired_out.data_ptr<float>(),
        t, Vrest, Vth, Vreset, refrac, decay, a_inc
    );

    snn_apply_stdp_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        plastic.data_ptr<float>(), fly_of_syn.data_ptr<int64_t>(), reward.data_ptr<float>(),
        fired_out.data_ptr<float>(), pre_trace.data_ptr<float>(), post_trace.data_ptr<float>(),
        eta_pot, eta_dep, dan_gain
    );

    snn_update_traces_kernel<<<blocks, threads>>>(
        N, pre_trace.data_ptr<float>(), post_trace.data_ptr<float>(),
        fired_out.data_ptr<float>(), pre_decay, post_decay
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step, "SNN forward step with continuous STDP (CUDA)");
}
