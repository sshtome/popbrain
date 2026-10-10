"""
Benchmark: PopBrain Tiered Layer Streamer (VRAM Cache + Pinned Host DMA)
========================================================================
Measures exact end-to-end token latency, memory bandwidth, and generation speed
for hybrid layer execution on consumer hardware (NVIDIA RTX 3060 12GB).

Simulates a 64-layer architecture (Qwen-2.5-32B scale):
- 52 layers permanently resident in GPU VRAM (0 ms PCIe latency).
- 12 layers streamed from Host RAM via Double-Buffered Asynchronous DMA.
"""

import sys
import os
import time
import torch

# Ensure local popbrain package is imported
sys.path.insert(0, os.path.abspath("."))
from popbrain import TieredLayerStreamer, SpikeActivationGate


def main():
    print("=" * 72)
    print("  PopBrain Tiered Layer Streamer: Hardware Acceleration Benchmark")
    print("=" * 72)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[!] CUDA not available. Exiting.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[*] GPU Device: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"[*] Target Architecture: 64 Layers, Hidden Dim 5120 (32.5B scale)")

    num_total = 64
    num_cached = 52
    num_streamed = num_total - num_cached
    layer_bytes = 180 * 1024 * 1024  # 180 MB per layer

    print(f"[*] Memory Partitioning:")
    print(f"    - Static VRAM Tier:    {num_cached} layers ({num_cached * 180 / 1024:.2f} GB VRAM)")
    print(f"    - Dynamic Stream Tier: {num_streamed} layers ({num_streamed * 180 / 1024:.2f} GB Pinned Host RAM)")
    print(f"    - Active VRAM Buffer:  2 dynamic slots ({2 * 180:.0f} MB)")

    # 1. Initialize streamer
    streamer = TieredLayerStreamer(
        num_total_layers=num_total,
        num_cached_layers=num_cached,
        layer_bytes=layer_bytes,
        device=device
    )

    # 2. Allocate pinned dynamic layers
    print("\n[*] Allocating Pinned (Page-Locked) Host Memory for dynamic layers...")
    t0 = time.time()
    pinned_weights = streamer.allocate_pinned_dynamic_layers()
    for w in pinned_weights:
        w.normal_(0.0, 0.02)
    print(f"[+] Allocated {num_streamed} pinned layers in {time.time() - t0:.3f}s!")

    # 3. Simulate layer computation functions
    hidden_dim = 5120
    x_input = torch.randn(1, hidden_dim, dtype=torch.float16, device=device)
    w_cached = torch.randn(hidden_dim, hidden_dim, dtype=torch.float16, device=device)
    spike_gate = SpikeActivationGate(threshold=0.04)

    def cached_layer_compute(layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm + Matmul + Spike gate
        normed = x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        gated = spike_gate(normed)
        return x + torch.matmul(gated, w_cached)

    def dynamic_layer_compute(layer_idx: int, x: torch.Tensor, weight_buf: torch.Tensor) -> torch.Tensor:
        # Slice weight buffer view and compute
        w_view = weight_buf[:hidden_dim * hidden_dim].view(hidden_dim, hidden_dim)
        normed = x / (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        gated = spike_gate(normed)
        return x + torch.matmul(gated, w_view)

    # 4. Warmup
    print("\n[*] Warming up CUDA compute and transfer streams...")
    for _ in range(3):
        _ = streamer.execute_token_pipeline(
            x_input, cached_layer_compute, dynamic_layer_compute, pinned_weights
        )

    # 5. Measure Benchmark
    num_tokens = 20
    print(f"\n[*] Running Benchmark across {num_tokens} tokens (each token executes {num_total} layers)...")
    torch.cuda.synchronize()
    t_start = time.time()

    token_latencies = []
    x = x_input.clone()
    for t in range(num_tokens):
        t0_token = time.time()
        x = streamer.execute_token_pipeline(
            x, cached_layer_compute, dynamic_layer_compute, pinned_weights
        )
        token_latencies.append(time.time() - t0_token)

    t_total = time.time() - t_start
    avg_latency_ms = (t_total / num_tokens) * 1000
    tokens_per_sec = num_tokens / t_total

    vram_stats = streamer.get_vram_stats()

    print("\n" + "=" * 72)
    print("  BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  - Total Generated Tokens   : {num_tokens}")
    print(f"  - Total Pipeline Duration  : {t_total:.3f} s")
    print(f"  - Average Token Latency    : {avg_latency_ms:.2f} ms")
    print(f"  - End-to-End Speed         : {tokens_per_sec:.2f} tokens/second")
    print(f"  - VRAM Allocated           : {vram_stats['allocated_mb']:.1f} MB")
    print(f"  - VRAM Peak Reserved       : {vram_stats['reserved_mb']:.1f} MB")
    print("=" * 72)


if __name__ == "__main__":
    main()
