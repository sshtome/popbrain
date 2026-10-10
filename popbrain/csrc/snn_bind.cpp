#include <torch/extension.h>

// Forward declarations of CUDA kernels
void snn_update_neurons_kernel(
    const int N, const int64_t* crow, const int64_t* col_idx, const float* W, const float* spikes_in,
    float* V, float* a, float* refrac_until, float* pre_trace, float* post_trace, float* fired_out,
    const float t, const float Vrest, const float Vth, const float Vreset, const float refrac,
    const float decay, const float a_inc, const float pre_decay, const float post_decay);

void snn_apply_stdp_kernel(
    const int N, const int64_t* crow, const int64_t* col_idx, float* W,
    const float* plastic, const int64_t* fly_of_syn, const float* reward,
    const float* fired, const float* pre_trace, const float* post_trace,
    const float eta_pot, const float eta_dep, const float dan_gain);


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

    // Launch Neuron Update Kernel
    snn_update_neurons_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        spikes_in.data_ptr<float>(), V.data_ptr<float>(), a.data_ptr<float>(),
        refrac_until.data_ptr<float>(), pre_trace.data_ptr<float>(),
        post_trace.data_ptr<float>(), fired_out.data_ptr<float>(),
        t, Vrest, Vth, Vreset, refrac, decay, a_inc, pre_decay, post_decay
    );

    // Launch Continuous STDP Kernel
    snn_apply_stdp_kernel<<<blocks, threads>>>(
        N, crow.data_ptr<int64_t>(), col_idx.data_ptr<int64_t>(), W.data_ptr<float>(),
        plastic.data_ptr<float>(), fly_of_syn.data_ptr<int64_t>(), reward.data_ptr<float>(),
        fired_out.data_ptr<float>(), pre_trace.data_ptr<float>(), post_trace.data_ptr<float>(),
        eta_pot, eta_dep, dan_gain
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_step", &forward_step, "SNN forward step with continuous STDP (CUDA)");
}
