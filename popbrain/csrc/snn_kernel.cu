#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * ============================================================================
 * Kernel 1: Neuron State Update, Sparse Synaptic Integration, and Spiking
 * ============================================================================
 *
 * Each CUDA thread is mapped to exactly one post-synaptic neuron i in the unified
 * population state vector (total size N = n_slots * N_neurons_per_agent).
 *
 * Algorithm per thread (neuron i):
 *   1. Sparse Matrix-Vector Product (SpMV):
 *      Iterates over incoming synapses defined by CSR row slice [crow[i], crow[i+1]).
 *      Accumulates weighted spikes: v_add = sum_e( W[e] * spikes_in[col_idx[e]] )
 *   2. Leaky Membrane Potential Integration:
 *      v(t+dt) = Vrest + (v(t) + v_add - Vrest) * decay
 *   3. Dynamic Thresholding & Refractory Check:
 *      A neuron fires if (v - a[i] >= Vth) AND current time t >= refrac_until[i].
 *      Here, a[i] is the spike-rate adaptation (After-Hyperpolarization / AHP).
 *   4. Spike Emission:
 *      If fired:
 *        - Reset voltage: v = Vreset
 *        - Set absolute refractory timeout: refrac_until[i] = t + refrac
 *        - Increase adaptation potential: a[i] += a_inc
 *        - Write spike output: fired_out[i] = 1.0f
 */
__global__ void snn_update_neurons_kernel(
    const int N,
    const int64_t* __restrict__ crow,          // CSR row pointers (size N + 1)
    const int64_t* __restrict__ col_idx,       // CSR column indices (pre-synaptic neuron IDs)
    const float* __restrict__ W,               // Synaptic weights array
    const float* __restrict__ spikes_in,       // Delayed incoming spikes from ring buffer
    float* __restrict__ V,                     // Membrane potential array [mV]
    float* __restrict__ a,                     // Spike-rate adaptation potential array [mV]
    float* __restrict__ refrac_until,          // Absolute refractory period expiration timestamps [ms]
    float* __restrict__ fired_out,             // Spike output array (1.0 = spike, 0.0 = silent)
    const float t,                             // Current simulation time [ms]
    const float Vrest,                         // Resting membrane potential [mV] (e.g. -65.0)
    const float Vth,                           // Base threshold [mV] (e.g. -59.0)
    const float Vreset,                        // Reset potential [mV] (e.g. -70.0)
    const float refrac,                        // Refractory duration [ms] (e.g. 3.0)
    const float decay,                         // Membrane leak decay factor: exp(-dt / tau)
    const float a_inc                          // Adaptation increment per spike [mV] (e.g. 2.5)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 1. Integrate incoming synaptic inputs from pre-synaptic neurons
    float v_add = 0.0f;
    int64_t start = crow[i];
    int64_t end = crow[i+1];
    for (int64_t e = start; e < end; ++e) {
        int64_t j = col_idx[e];
        v_add += W[e] * spikes_in[j];
    }

    // 2. Leaky integration of membrane potential
    float v = V[i];
    v = Vrest + (v + v_add - Vrest) * decay;

    // 3. Spike generation with spike-frequency adaptation (AHP)
    bool fired = (v - a[i] >= Vth) && (t >= refrac_until[i]);
    float fired_f = fired ? 1.0f : 0.0f;

    // 4. Handle refractory period and reset
    if (fired) {
        v = Vreset;
        refrac_until[i] = t + refrac;
        a[i] += a_inc;
    }

    V[i] = v;
    fired_out[i] = fired_f;
}

/**
 * ============================================================================
 * Kernel 2: Three-Factor Spike-Timing-Dependent Plasticity (STDP)
 * ============================================================================
 *
 * Updates synaptic weights W[e] based on timing correlations between pre-
 * and post-synaptic spikes, modulated by individual organism rewards (Dopamine).
 *
 * Critical Order of Operations:
 *   Runs using the eligibility traces from the PREVIOUS time step, matching
 *   biological causality:
 *   - Post-synaptic spike at t binds with PRE-synaptic trace -> LTP (Potentiation)
 *   - Pre-synaptic spike at t binds with POST-synaptic trace -> LTD (Depression)
 *
 * Neuromodulation (Dopamine):
 *   mod = 1.0 + dan_gain * reward[fly] * plastic[e]
 *   Where:
 *     - reward[fly] is in [-1.0, 1.0] (organism's global dopamine signal)
 *     - plastic[e] is 1.0 for plastic neuropils (e.g. Mushroom Body), 0.0 otherwise
 */
__global__ void snn_apply_stdp_kernel(
    const int N,
    const int64_t* __restrict__ crow,          // CSR row pointers
    const int64_t* __restrict__ col_idx,       // CSR column indices
    float* __restrict__ W,                     // Synaptic weights (updated in-place)
    const float* __restrict__ plastic,         // Plasticity mask per synapse (1.0 = plastic, 0.0 = static)
    const int64_t* __restrict__ fly_of_syn,    // Mapping from synapse index to organism index (0..n_slots-1)
    const float* __restrict__ reward,          // Global reward / dopamine array per organism
    const float* __restrict__ fired,           // Spikes emitted in Kernel 1
    const float* __restrict__ pre_trace,       // Pre-synaptic eligibility traces from step t-1
    const float* __restrict__ post_trace,      // Post-synaptic eligibility traces from step t-1
    const float eta_pot,                       // Potentiation learning rate (LTP)
    const float eta_dep,                       // Depression learning rate (LTD)
    const float dan_gain                       // Dopamine gain multiplier
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
        
        // Skip if neither pre nor post neuron spiked this step
        if (pre_fired == 0.0f && post_fired == 0.0f) continue;
        
        float pre_tr = pre_trace[j];
        
        // Biological STDP equation:
        // LTP: Post fires -> strengthen connection proportional to pre trace
        // LTD: Pre fires -> weaken connection proportional to post trace
        float dw = post_fired * (pre_tr * eta_pot) - pre_fired * (post_tr * eta_dep);
        
        if (dw != 0.0f) {
            // Apply 3-factor dopamine modulation
            float mod = 1.0f + dan_gain * reward[fly_of_syn[e]] * plastic[e];
            W[e] += dw * mod;
        }
    }
}

/**
 * ============================================================================
 * Kernel 3: Eligibility Traces Leaky Update
 * ============================================================================
 *
 * Updates exponential eligibility traces for all neurons AFTER Kernel 2 executes:
 *   trace(t+dt) = (trace(t) + spike(t)) * decay
 */
__global__ void snn_update_traces_kernel(
    const int N,
    float* __restrict__ pre_trace,             // Pre-synaptic eligibility traces
    float* __restrict__ post_trace,            // Post-synaptic eligibility traces
    const float* __restrict__ fired_out,       // Newly emitted spikes
    const float pre_decay,                     // Pre-trace leak decay: exp(-dt / tau_pre)
    const float post_decay                     // Post-trace leak decay: exp(-dt / tau_post)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float fired_f = fired_out[i];
    pre_trace[i] = (pre_trace[i] + fired_f) * pre_decay;
    post_trace[i] = (post_trace[i] + fired_f) * post_decay;
}

/**
 * ============================================================================
 * PyTorch C++ Extension Wrapper: forward_step
 * ============================================================================
 *
 * Orchestrates the sequential launch of all 3 CUDA kernels on the active stream.
 * Invoked directly from Python via `popbrain.snn_cuda.forward_step(...)`.
 */
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

    // 1. Update voltages, integrate sparse inputs, check thresholds, fire spikes
    snn_update_neurons_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        spikes_in.data_ptr<float>(), V.data_ptr<float>(), a.data_ptr<float>(),
        refrac_until.data_ptr<float>(), fired_out.data_ptr<float>(),
        t, Vrest, Vth, Vreset, refrac, decay, a_inc
    );

    // 2. Compute three-factor STDP synaptic weight updates
    snn_apply_stdp_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        plastic.data_ptr<float>(), fly_of_syn.data_ptr<int64_t>(), reward.data_ptr<float>(),
        fired_out.data_ptr<float>(), pre_trace.data_ptr<float>(), post_trace.data_ptr<float>(),
        eta_pot, eta_dep, dan_gain
    );

    // 3. Update leaky pre/post eligibility traces for the next step
    snn_update_traces_kernel<<<blocks, threads>>>(
        N, pre_trace.data_ptr<float>(), post_trace.data_ptr<float>(),
        fired_out.data_ptr<float>(), pre_decay, post_decay
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step, "SNN forward step with continuous STDP (CUDA)");
}
