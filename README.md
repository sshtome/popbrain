# PopBrain 🧠⚡

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

PopBrain requires a CUDA-capable GPU and PyTorch.

```bash
git clone https://github.com/sshtome/popbrain.git
cd popbrain
pip install -e .
```

*Note: PopBrain uses JIT compilation for its CUDA extensions, which means it will compile the kernels automatically on the first import. Ensure you have the MSVC/CUDA compiler available in your terminal (e.g., Developer Command Prompt on Windows).*

## Quick Start

### 1. Define your Connectome
PopBrain expects the brain topology to be defined in a JSON format containing the neurons and the sparse synaptic connections (`p` -> `q` with weight `w`).

```json
{
  "neurons": [
    {
      "zone": "sensory_vision",
      "nt": "Glu",
      "role": "receptor"
    },
    {
      "zone": "motor_leg",
      "nt": "GABA",
      "role": "actuator"
    }
  ],
  "syn": {
    "p": [0],
    "q": [1],
    "w": [0.5]
  }
}
```

### 2. Run the Simulation
Once you have the JSON, initializing and running the simulation is extremely simple using input channels and output pools:

```python
import torch
import json
from popbrain import PopBrain

# Initialize the brain engine for 32 parallel agents (slots)
brain = PopBrain(
    n_slots=32,
    minibrain_path="my_connectome.json"
)

# Define an input channel for sensory neurons (e.g., neuron index 0)
# and an output pool for motor neurons (e.g., neuron index 1)
for agent_id in range(32):
    brain.add_channel("sensor_x", idxs=[0], rate=0.0, w=10.0, fly=agent_id)
    brain.add_pool("motor_y", idxs=[1], fly=agent_id)

brain.commit_channels()

# Simulation Loop
for step in range(100):
    # Set the firing rate (Hz) for the sensor of agent 0
    brain.set_rate("sensor_x", rate=150.0, fly=0)
    
    # Run the continuous-time simulation for 20 milliseconds
    brain.step(ms=20.0)
    
    # Read the integrated motor output rate (Hz) for agent 0
    motor_activity = brain.pool_rate("motor_y", fly=0)
    
    # Apply global reward/dopamine (modulates STDP plasticity)
    brain.reward_signal(amount=1.5, fly=0)
```

## Built-in ES Optimizer (Neuroevolution)

Training Spiking Neural Networks via backpropagation is notoriously difficult. PopBrain includes a built-in **OpenAI Evolution Strategies (ES)** optimizer, accelerated by Adam, Antithetic Sampling, and Rank Normalization. It is specifically designed to work seamlessly with PopBrain's parallel batches.

```python
from popbrain.optim import AdamES

# Requires an odd number of agents (1 champion + N even clones for antithetic noise)
brain = PopBrain(n_slots=25, minibrain_path="my_connectome.json")

# Initialize optimizer
optimizer = AdamES(brain, lr=0.1, sigma=0.05)

for epoch in range(100):
    # 1. Perturb clone weights automatically
    optimizer.ask(champ_fly=0)
    
    # 2. Run your simulation...
    brain.step(ms=100.0)
    
    # 3. Calculate rewards for all 25 agents
    rewards = get_rewards() # shape: (25,)
    
    # 4. Compute gradient and update the Champion
    optimizer.tell(rewards)
```

## Architecture

PopBrain separates the **topology** from the **simulation**. 
The topology is defined by your connectome data (which neurons exist, their neurotransmitter types, and the sparse connectivity graph).
The simulation engine (`PopBrain`) compiles this topology into a highly optimized memory layout (allocating block-diagonal Sparse CSR matrices for batch processing) and dispatches the actual voltage updates to the `snn_cuda` extension.

## License

MIT License. See `LICENSE` for more information.
