"""
Example: Running Microsoft BitNet-b1.58-2B-4T with PopBrain
===========================================================
Executes a 2.0 Billion parameter Ternary Large Language Model (weights in {-1, 0, +1})
on consumer GPU with zero matrix multiplications (GEMM-free additions/subtractions).

Expected Performance (NVIDIA RTX 3060 12GB):
- Parameters : 2.0 Billion (30 layers, 2560 hidden dimension)
- Memory     : ~4.5 GB VRAM (from 12 GB available)
- Speed      : ~20 tokens / second
"""

import os
import sys
import time
import torch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from popbrain.models import BitNet2B


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "d:/projects/models/bitnet-b1.58-2B-4T"

    print("=" * 65)
    print("  PopBrain BitNet-b1.58-2B-4T Inference Demo")
    print(f"  Device: {device.upper()} | Synaptic Type: 1.58-bit Ternary {{-1, 0, +1}}")
    print("=" * 65)

    if not os.path.exists(model_dir):
        print(f"[!] Error: Model directory not found at {model_dir}")
        print("    Download from: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T")
        return

    # 1. Load model onto GPU
    print("[*] Loading BitNet-2B weights into CUDA...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    bitnet = BitNet2B(model_dir=model_dir, device=device, dtype=torch.bfloat16)
    load_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"[+] Model loaded in {load_time:.2f}s | VRAM: {peak_vram:.1f} MB ({peak_vram/1024:.2f} GB)")

    # 2. Generate text
    prompt = "Artificial intelligence and brain-inspired spiking networks"
    print(f"\nPrompt: \"{prompt}\"")
    print("[*] Generating text...")
    
    t_gen_start = time.time()
    response = bitnet.generate(
        prompt=prompt,
        max_new_tokens=40,
        temperature=0.7,
        top_p=0.9
    )
    gen_time = time.time() - t_gen_start

    print("\n" + "-" * 65)
    print(prompt + response)
    print("-" * 65)

    peak_vram_final = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"\n[+] Generation completed in {gen_time:.2f}s")
    print(f"[+] Peak VRAM: {peak_vram_final:.1f} MB ({peak_vram_final/1024:.2f} GB)")


if __name__ == "__main__":
    main()
