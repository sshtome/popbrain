import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
import time
from popbrain.dopamine.scalable_memory import PopBrainScale10kMemory

def main():
    print("Testing PopBrainScale10kMemory on CUDA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 2560
    num_transitions = 160000

    mem = PopBrainScale10kMemory(
        hidden_dim=hidden_dim,
        num_clusters=256,
        nprobe=4,
        inhibition_threshold=0.85,
        margin_threshold=0.02,
        modulation_strength=0.18,
        device=device
    )

    # 1. Bulk ingestion of synthetic normalized keys and values
    t0 = time.time()
    keys = torch.randn(num_transitions, hidden_dim, device=device, dtype=torch.bfloat16)
    keys = keys / torch.norm(keys, dim=-1, keepdim=True)
    vals = torch.randn(num_transitions, hidden_dim, device=device, dtype=torch.bfloat16)
    vals = vals / torch.norm(vals, dim=-1, keepdim=True)

    mem.ingest_batch(keys, vals)
    print(f"Ingested {num_transitions} transitions in {time.time()-t0:.2f}s.")

    # 2. Build IVF Index
    t0 = time.time()
    mem.build_ivf_index(iterations=3)
    print(f"Built IVF-256 Index in {time.time()-t0:.3f}s. IVF built: {mem.ivf_built}")

    # 3. Benchmark query retrieval latency
    query = keys[42].clone() # Target query (should fire)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        delta_h, max_sim = mem.compute_synaptic_modulation(query)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 100 * 1000

    print(f"Retrieval latency per query: {dt:.3f} ms")
    print(f"Max similarity observed: {max_sim:.4f}")
    print(f"Delta h norm: {delta_h.norm().item():.4f} (Fired: {delta_h.norm().item() > 0})")

    # 4. Check distractor query (should NOT fire)
    distractor = torch.randn(hidden_dim, device=device, dtype=torch.bfloat16)
    distractor = distractor / torch.norm(distractor)
    delta_h_dist, max_sim_dist = mem.compute_synaptic_modulation(distractor)
    print(f"Distractor max sim: {max_sim_dist:.4f}")
    print(f"Distractor fired: {delta_h_dist.norm().item() > 0}")

    print("PopBrainScale10kMemory self-test PASSED successfully!")

if __name__ == "__main__":
    main()
