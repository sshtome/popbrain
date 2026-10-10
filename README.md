# PopBrain рџ§ вљЎ

PopBrain is a highly optimized, high-performance **Spiking Neural Network (SNN)** engine for PyTorch. 
It is designed to simulate extremely large, sparse neural connectomes (hundreds of thousands of synapses) in parallel batches, making it ideal for Reinforcement Learning (RL), evolutionary strategies, and robotics.

Under the hood, it uses **PyTorch Sparse CSR matrices** and custom **CUDA kernels** to deliver millisecond-level inference for massive populations of agents.

## Features

- **Extreme Performance:** Simulates massive networks using custom CUDA kernels and Sparse CSR tensors.
- **Population Batching:** Simulate $N$ parallel "clones" of a brain simultaneously (e.g., for ES or RL vector environments).
- **Continuous STDP:** Built-in Spike-Timing-Dependent Plasticity. Synapses learn continuously based on spike timings.
- **Neuromodulation:** Supports global reward signals (Dopamine) to modulate plasticity dynamically.
- **E/I Balance:** Enforces Dale's Law natively (excitatory neurons stay excitatory, inhibitory stay inhibitory).
- **Continuous-Time Simulation:** Define continuous Poisson input channels and read integrated output rates via spike pooling.

## Installation


