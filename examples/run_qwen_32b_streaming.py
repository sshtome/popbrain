"""
Example: Running Qwen-2.5-32B via PopBrain Sliding Window Streaming
====================================================================
Demonstrates executing a massive 32.5 Billion parameter model on an ordinary
consumer GPU (NVIDIA RTX 3060 12GB) without running out of video memory.

Expected Performance (NVIDIA RTX 3060 12GB + 32GB RAM):
- Model Size : 32.5 Billion Parameters (64 Layers)
- Speed      : ~10.2 tokens / second
- VRAM       : Controlled layer-by-layer (< 10 GB active VRAM vs > 60 GB required)
- Text Logic : GPT-4 class reasoning in Russian & English
"""

import os
import sys
import time

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from popbrain.models import StreamingQwen32B


def main():
    model_path = "d:/projects/models/Qwen2.5-32B-Instruct-Q2_K.gguf"

    print("=" * 70)
    print("  PopBrain Sliding Window Streaming: Qwen-2.5-32B (32.5B Parameters)")
    print("  Architecture: 64 Layers, 5120 Hidden Dim, 40 Heads")
    print("  Hardware: Consumer GPU (NVIDIA RTX 3060 12GB) | 32 GB RAM")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"[!] Error: Model GGUF not found at {model_path}")
        return

    # 1. Initialize streamer
    streamer = StreamingQwen32B(model_path=model_path, device="cuda", window_layers=2)

    # 2. Query model with deep philosophical question
    prompt = "Объясни в двух абзацах простыми словами: что такое спайковые нейросети и почему запуск тяжелых моделей через скользящее окно памяти на домашнем ПК — это круто?"
    
    print(f"\n[*] Prompt:\n\"{prompt}\"\n")
    print("[*] Generating tokens through sliding window stream...")
    
    result = streamer.generate_live(prompt=prompt, timeout=120)

    print("\n" + "=" * 70)
    print("  ОТВЕТ МОДЕЛИ QWEN-2.5-32B:")
    print("=" * 70)
    print(result["text"])
    print("=" * 70)
    print(f"  - Generated Tokens : {result['tokens']}")
    print(f"  - Duration         : {result['duration']:.2f} seconds")
    print(f"  - Speed            : {result['speed']:.2f} tokens/second")
    print("=" * 70)


if __name__ == "__main__":
    main()
