# PopBrain Engine: Internal Architecture & Technical Guide

PopBrain is a high-performance Spiking Neural Network (SNN) engine engineered specifically for simulating **large-scale biological connectomes** ($25,000+$ neurons, hundreds of thousands of synapses) in parallel across populations of reinforcement learning and evolutionary agents.

This document serves as an exhaustive technical guide to how PopBrain works under the hood.

---

## 1. High-Level Design & Core Tenets

Traditional deep learning SNN frameworks (e.g. `snnTorch`, `SpikingJelly`) treat SNNs as standard feedforward/recurrent neural networks with dense weights (`torch.nn.Linear`). For real biological brains, this approach fails:
- **Biological connectivity is extremely sparse** (~0.05% non-zero connections).
- Storing a $26,000$-neuron brain as a dense matrix requires $26,000^2 \times 4\text{ bytes} \approx 2.7\text{ GB}$ per organism.
- Simulating 32 parallel mutant agents in an evolutionary loop would require **$>80\text{ GB}$ VRAM**, crashing consumer GPUs.

**PopBrain solves this through four core pillars:**
1. **Block-Diagonal Sparse CSR Representation**: Packs $N$ independent agent brains into a single unified Sparse CSR matrix on the GPU, achieving zero inter-agent communication overhead and $170\times$ memory reduction.
2. **Circular Ring Buffers for Axonal Delays**: Models realistic $2\text{ ms}$ synaptic propagation delays without duplicating tensors.
3. **Custom CUDA Kernels (`snn_kernel.cu`)**: Fuses the sparse matrix-vector multiplication (SpMV), membrane potential decay, thresholding, spike generation, and refractory updates into single-pass GPU kernels.
4. **Three-Factor Continuous STDP with Dopamine Neuromodulation**: Synaptic weights adapt in real time during physics simulation, modulated by an organism's reward signal.

---

## 2. Mathematical Model

### A. Leaky Integrate-and-Fire (LIF) Neuron
Each neuron $i$ has membrane potential $V_i(t)$. In discrete time with step $dt$, the update equation is:

$$V_i(t + dt) = V_{\text{rest}} + \left( V_i(t) + I_{\text{syn}, i}(t) - V_{\text{rest}} \right) \cdot \exp\left(-\frac{dt}{\tau}\right)$$

Where incoming synaptic current $I_{\text{syn}, i}$ is the sum of weighted spikes from pre-synaptic neurons $j$:

$$I_{\text{syn}, i}(t) = \sum_{j \in \text{pre}(i)} W_{ij} \cdot S_j(t - \text{delay})$$

### B. Spike Generation & Spike-Frequency Adaptation (AHP)
A neuron fires a spike if its membrane potential exceeds its effective threshold and it is outside its absolute refractory period:

$$S_i(t) = 1 \quad \text{if} \quad V_i(t) - a_i(t) \ge V_{\text{th}} \quad \text{and} \quad t \ge t_{\text{refrac}, i}$$

When a spike occurs:
1. Membrane potential resets: $V_i(t) \leftarrow V_{\text{reset}}$.
2. Refractory counter is set: $t_{\text{refrac}, i} \leftarrow t + \tau_{\text{refrac}}$.
3. Adaptation potential (After-Hyperpolarization, AHP) increases: $a_i \leftarrow a_i + a_{\text{inc}}$.

Between spikes, adaptation decays exponentially:

$$a_i(t + dt) = a_i(t) \cdot \exp\left(-\frac{dt}{\tau_a}\right)$$

### C. Dale's Law Enforcement (E/I Balance)
In biology, a neuron cannot switch from excitatory to inhibitory. PopBrain natively enforces this rule:
- Excitatory neurons (e.g. Acetylcholine `ACH`, Glutamate `GLU`): $W_{ij} \ge 0$, clamped to $[0, 2 W_{0, ij}]$.
- Inhibitory neurons (e.g. GABA `GABA`): $W_{ij} \le 0$, clamped to $[2 W_{0, ij}, 0]$.

### D. Three-Factor STDP with Dopamine Neuromodulation
Synapses track exponential eligibility traces for pre- and post-synaptic spikes:

$$\text{trace}_{\text{pre}, j}(t + dt) = (\text{trace}_{\text{pre}, j}(t) + S_j(t)) \cdot \exp\left(-\frac{dt}{\tau_{\text{pre}}}\right)$$
$$\text{trace}_{\text{post}, i}(t + dt) = (\text{trace}_{\text{post}, i}(t) + S_i(t)) \cdot \exp\left(-\frac{dt}{\tau_{\text{post}}}\right)$$

The weight update $\Delta W_{ij}$ is computed as:

$$\Delta W_{ij} = \left[ S_i(t) \cdot \text{trace}_{\text{pre}, j}(t) \cdot \eta_{\text{pot}} - S_j(t) \cdot \text{trace}_{\text{post}, i}(t) \cdot \eta_{\text{dep}} \right] \cdot \left( 1 + \text{gain} \cdot R_k \cdot P_{ij} \right)$$

Where:
- $\eta_{\text{pot}}, \eta_{\text{dep}}$ are LTP and LTD learning rates.
- $R_k \in [-1.0, 1.0]$ is the global Dopamine reward signal of agent $k$.
- $P_{ij} \in \{0, 1\}$ is the structural plasticity mask (e.g., Mushroom Body learning zones).

---

## 3. GPU Memory Layout: Block-Diagonal Sparse CSR

To simulate $K$ parallel agents without inter-agent crosstalk and without looping in Python:

```
Unified State Vector (Length = K * N_neurons):
[ Agent 0 Neurons | Agent 1 Neurons | ... | Agent K-1 Neurons ]

Unified Sparse CSR Matrix (K * N_neurons x K * N_neurons):
┌───────────┬───────────┬─────┬───────────┐
│ Brain 0   │     0     │ ... │     0     │
├───────────┼───────────┼─────┼───────────┤
│     0     │  Brain 1  │ ... │     0     │
├───────────┼───────────┼─────┼───────────┤
│    ...    │    ...    │ ... │    ...    │
├───────────┼───────────┼─────┼───────────┤
│     0     │     0     │ ... │ Brain K-1 │
└───────────┴───────────┴─────┴───────────┘
```

1. **`crow` (Compressed Row Pointers)**: Points to the start and end of outgoing synapses for each neuron in CSR format.
2. **`col_idx` (Column Indices)**: Target pre-synaptic neuron index. For agent $k$, indices are offset by $k \cdot N_{\text{neurons}}$, guaranteeing zero memory access outside the agent's slice.
3. **`W` (Dynamic Synaptic Weights)**: Continuously updated by STDP and weight decay.
4. **`W0` (Baseline Genotype Weights)**: Base weights mutated by evolutionary strategies.

---

## 4. Synaptic Delay Pipeline (Circular Ring Buffer)

Axons require time to conduct action potentials (~2 ms in insect brains). PopBrain avoids copying or queue allocations using a pre-allocated **Circular Ring Buffer**:

$$\text{Slot Count} = \left\lceil \frac{\text{delay}_{\text{ms}}}{dt} \right\rceil + 1$$

At each step:
1. **Read**: The incoming spikes are fetched from `spike_ring[step_idx % Slot Count]`.
2. **Execute**: The CUDA kernel updates voltages and generates new spikes $S(t)$.
3. **Write**: The new spikes are stored into `spike_ring[(step_idx + delay) % Slot Count]`.

This provides $O(1)$ constant time delay propagation with zero memory allocations during simulation.

---

## 5. CUDA Kernel Execution Flow

Inside `popbrain/csrc/snn_kernel.cu`, the `forward_step` function dispatches 3 sequential GPU kernels:

```
              ┌──────────────────────────────────────────────┐
              │ 1. snn_update_neurons_kernel                 │
              │ - SpMV inline dot product: sum(W * spikes)   │
              │ - Leaky voltage update: V = Vrest + (V)*decay│
              │ - Threshold comparison: (V - a >= Vth)       │
              │ - Refractory counter & spike output          │
              └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ 2. snn_apply_stdp_kernel                     │
              │ - Read OLD pre- and post-traces              │
              │ - Compute LTP/LTD: dw = post*pre_tr - pre*...│
              │ - Scale by Dopamine: mod = 1 + gain*R*P      │
              │ - In-place update: W[e] += dw * mod          │
              └──────────────────────┬───────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────────────┐
              │ 3. snn_update_traces_kernel                  │
              │ - Add new spikes to eligibility traces       │
              │ - Apply exponential decay: trace *= decay    │
              └──────────────────────────────────────────────┘
```

---

## 6. Channel I/O Pipeline (Sensory & Motor)

```
Environment Sensor
       │
       ▼ (Hz)
 [add_channel] ──► Pinned Host Memory ──► Asynchronous DMA ──► VRAM
                                                                 │
                                                                 ▼
                                                    Poisson Spike Injection
                                                                 │
                                                                 ▼
                                                          Connectome Step
                                                                 │
                                                                 ▼
 [pool_rate]   ◄── Integrated Spike Rate (Hz) ◄── Motor Pooling Mask
       │
       ▼
Motor Actuator
```

- **Pinned CPU Memory (`pin_memory=True`)**: Sensor rates written on CPU are streamed to GPU via non-blocking DMA copies (`copy_(..., non_blocking=True)`).
- **Spike Pooling**: Motor outputs read accumulated spikes across defined neuron groups and normalize them to firing rates (Hz) without CPU-GPU synchronization bottlenecks.
