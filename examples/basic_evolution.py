import torch
import json
import numpy as np
from popbrain import PopBrain
from popbrain.optim import AdamES

# 1. Dummy connectome
connectome_data = {
  "neurons": [
    {"zone": "sensory", "nt": "Glu", "role": "receptor"},
    {"zone": "motor", "nt": "GABA", "role": "actuator"}
  ],
  "syn": {"p": [0], "q": [1], "w": [0.5]}
}
with open("dummy_brain.json", "w") as f:
    json.dump(connectome_data, f)

# 2. Initialize Brain (must be odd number of slots for Antithetic ES e.g., 25)
N_AGENTS = 25
brain = PopBrain(n_slots=N_AGENTS, minibrain_path="dummy_brain.json")

for i in range(N_AGENTS):
    brain.add_channel("sensor", [0], 0.0, 10.0, i)
    brain.add_pool("motor", [1], i)
brain.commit_channels()

# 3. Initialize Optimizer
optimizer = AdamES(brain, lr=0.1, sigma=0.1)

# 4. Evolution Loop
print("Starting evolution...")
for epoch in range(10):
    # Step A: Perturb the weights of clones (Agent 0 remains pure champ)
    optimizer.ask(champ_fly=0)
    
    # Run simulation for all 25 agents
    brain.set_rate_by_index(0, 150.0) # sensor on
    brain.step(ms=20.0)
    
    # Step B: Evaluate fitness (e.g. we want motor to fire as much as possible)
    rewards = np.array([brain.pool_rate("motor", fly=i) for i in range(N_AGENTS)])
    
    # Step C: Compute gradient and update champion
    # We pass the full array; tell() automatically skips the champ in slot 0
    optimizer.tell(rewards)
    
    print(f"Epoch {epoch} | Champ reward: {rewards[0]:.2f}")

print("Evolution complete!")
