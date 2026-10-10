# PopBrain 🧠⚡

PopBrain is a highly optimized, high-performance **Spiking Neural Network (SNN)** engine for PyTorch. 
It is designed to simulate extremely large, sparse neural connectomes (hundreds of thousands of synapses) in parallel batches, making it ideal for Reinforcement Learning (RL), evolutionary strategies, and robotics.

Under the hood, it uses **PyTorch Sparse CSR matrices** and custom **CUDA kernels** to deliver millisecond-level inference for massive populations of agents.

## Features

- **Extreme Performance:** Simulates massive networks using custom CUDA kernels and Sparse CSR tensors.
- **Population Batching:** Simulate $ parallel "clones" of a brain simultaneously (e.g., for ES or RL vector environments).
- **Continuous STDP:** Built-in Spike-Timing-Dependent Plasticity. Synapses learn continuously based on spike timings.
- **Neuromodulation:** Supports global reward signals (Dopamine) to modulate plasticity dynamically.
- **E/I Balance:** Enforces Dale's Law natively (excitatory neurons stay excitatory, inhibitory stay inhibitory).
- **Environment Agnostic:** A pure mathematical engine. Feed it an input current tensor, get an output spike tensor.

## Installation

PopBrain requires a CUDA-capable GPU and PyTorch.

\\\ash
git clone https://github.com/sshtome/popbrain.git
cd popbrain
pip install -e .
\\\

*Note: The installation will compile the custom CUDA extensions. Ensure you have the CUDA toolkit and a compatible C++ compiler installed.*

## Quick Start

### 1. Define your Connectome
PopBrain expects the brain topology to be defined in a JSON format containing the neurons and the sparse synaptic connections (p -> q with weight w).

\\\json
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
\\\

### 2. Run the Simulation
Once you have the JSON, initializing and running the simulation is extremely simple:

\\\python
import torch
import json
from popbrain import PopBrain

# Load the topology
with open("my_connectome.json", "r") as f:
    connectome_data = json.load(f)

# Initialize the brain engine for 32 parallel agents
brain = PopBrain(
    connectome_data, 
    batch_size=32, 
    device='cuda'
)

# Simulation Loop
for step in range(1000):
    # Create input currents for the 32 agents (e.g., from environment sensors)
    # Shape: [batch_size, num_neurons]
    I = torch.zeros((32, len(connectome_data['neurons'])), device='cuda')
    
    # Inject current into specific sensory neurons
    I[:, 0] = 5.0 
    
    # Forward pass (1 millisecond of simulation)
    spikes = brain.forward(I)
    
    # Read motor spikes
    motor_activity = spikes[:, 1] # Check if neuron 1 spiked
    
    # Apply global reward/dopamine (modulates STDP)
    brain.apply_reward(torch.ones(32, device='cuda') * 0.1)

\\\

## Architecture

PopBrain separates the **topology** from the **simulation**. 
The topology is defined by your connectome data (which neurons exist, their neurotransmitter types, and the sparse connectivity graph).
The simulation engine (\PopBrain\) compiles this topology into a highly optimized memory layout (allocating block-diagonal Sparse CSR matrices for batch processing) and dispatches the actual voltage updates to the \snn_cuda\ extension.

## License

MIT License. See \LICENSE\ for more information.
