"""
Example: Zero-Backprop Wikipedia Factual Ingestion with PopBrain Synaptic Memory
================================================================================
Demonstrates in-flight acquisition of authentic Wikipedia articles into a 2.4B
parameter frozen language model without gradient backpropagation (0 autograd bytes).

Uses PopBrain Hetero-Associative Synaptic Memory with GABAergic Lateral Inhibition.

Usage:
------
python examples/run_wikipedia_synaptic_learning.py
"""

import os
import sys
import time
from typing import List

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from popbrain import PopBrainSynapticMemory


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "D:/projects/models/bitnet-b1.58-2B-4T"

    print("=" * 80)
    print("  PopBrain Zero-Backprop Wikipedia Synaptic Learning Demo")
    print(f"  Model: BitNet-b1.58-2B-4T | Device: {device.upper()}")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"[!] Error: Model not found at {model_path}")
        return

    # 1. Load frozen LLM
    print("\n[*] Loading BitNet-2B into VRAM...")
    tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()

    embed_layer = model.get_input_embeddings()
    hidden_dim = model.config.hidden_size

    # 2. Initialize Synaptic Memory
    synaptic_memory = PopBrainSynapticMemory(
        hidden_dim=hidden_dim,
        inhibition_threshold=0.85,
        modulation_strength=0.15,
        device=device
    )

    # 3. Target Wikipedia Article Excerpt
    blind_prompt = "The Vera C. Rubin Observatory is located on the mountain of"
    target_continuation = " Cerro Pachon in northern Chile, at an altitude of 2,682 meters."
    full_wikipedia_text = blind_prompt + target_continuation

    print(f"\n[Prompt]: \"{blind_prompt}\"")
    print(f"[Target Fact]: \"{target_continuation.strip()}\"")

    # 4. Baseline Generation (Before Training)
    print("\n[*] Baseline Generation (Pre-Training Zero-Shot)...")
    prompt_ids = tok(blind_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out_base = model.generate(prompt_ids, max_new_tokens=15, do_sample=False)
    base_text = tok.decode(out_base[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"[-] Baseline Model Generated: \"{base_text}\"")

    # 5. Ingest Wikipedia Fact via Three-Factor Hebbian Plasticity
    print("\n[*] Ingesting Wikipedia Text into Synaptic Memory (Zero-Backprop)...")
    t0 = time.time()
    seq_ids = tok(full_wikipedia_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]

    with torch.no_grad():
        out = model(seq_ids.unsqueeze(0), output_hidden_states=True)
        h_seq = out.hidden_states[-1][0]
        e_seq = embed_layer(seq_ids)

        synaptic_memory.ingest_sequence(h_seq, e_seq, reward=1.0)

    duration_ms = (time.time() - t0) * 1000
    print(f"[+] Ingested {synaptic_memory.num_memories} synaptic transitions in {duration_ms:.1f} ms.")

    # 6. Post-Training Blind Generation (Modulated by Synaptic Memory)
    print("\n[*] Post-Training Generation with Synaptic Modulation...")
    curr_ids = tok(blind_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
    gen_tokens = []

    for _ in range(15):
        with torch.no_grad():
            out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
            h_last = out.hidden_states[-1][0, -1, :]

            delta_h, max_sim = synaptic_memory.compute_synaptic_modulation(h_last)
            if delta_h.norm() > 0:
                h_mod = h_last + delta_h
                logits = model.lm_head(h_mod.unsqueeze(0))[0]
            else:
                logits = out.logits[0, -1, :]

            next_id = int(torch.argmax(logits).item())
            curr_ids = torch.cat([curr_ids, torch.tensor([next_id], device=device)])
            gen_tokens.append(next_id)

            if next_id == tok.eos_token_id or tok.decode([next_id]).strip() == ".":
                break

    post_text = tok.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"[+] Post-Training Synaptic Output: \"{post_text}\"")

    print("\n" + "=" * 80)
    print("  SUMMARY:")
    print(f"  - Baseline Hallucination : \"{base_text}\"")
    print(f"  - Post-Synaptic Recall   : \"{post_text}\"")
    print(f"  - Ingestion Latency      : {duration_ms:.1f} ms")
    print(f"  - Backpropagation Graph  : 0 bytes (torch.no_grad)")
    print("=" * 80)


if __name__ == "__main__":
    main()
