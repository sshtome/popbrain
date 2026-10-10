import torch
import json
from popbrain import PopBrain

# Simple connectome
connectome_data = {
  "neurons": [
    {"zone": "sensory", "nt": "Glu", "role": "receptor"},
    {"zone": "motor", "nt": "GABA", "role": "actuator"}
  ],
  "syn": {
    "p": [0],
    "q": [1],
    "w": [0.5]
  }
}

with open("tests/test_connectome.json", "w") as f:
    json.dump(connectome_data, f)

print("Initializing PopBrain...")
brain = PopBrain(
    n_slots=2,
    minibrain_path="tests/test_connectome.npz"
)

for i in range(2):
    brain.add_channel("sensor", [0], 0.0, 10.0, i)
    brain.add_pool("motor", [1], i)

brain.commit_channels()
print("Brain initialized and channels committed successfully!")

brain.set_rate("sensor", 150.0, fly=0)
brain.step(ms=20.0)
rate = brain.pool_rate("motor", 0)

print(f"Motor output rate after 20ms: {rate:.2f} Hz")
print("Engine test complete!")


