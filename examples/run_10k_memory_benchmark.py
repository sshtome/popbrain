"""
Benchmark: PopBrain 10,000 Facts Synaptic Memory Scaling & Cross-Talk Isolation
================================================================================
Scientific benchmark verifying retention, interference, and retrieval latency
at massive scale (up to 10,000 factual associations) without backpropagation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import json
from typing import List, Dict, Tuple, Any

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from popbrain.dopamine.scalable_memory import PopBrainScale10kMemory


def load_dataset(dataset_path: str = "assets/datasets/squad_10k.json") -> List[Dict[str, str]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark():
    print("=" * 80)
    print("PopBrain 10,000 Facts Synaptic Memory Scaling Benchmark")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "D:/projects/models/bitnet-b1.58-2B-4T"

    print("Loading BitNet-b1.58-2B-4T on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    embed_layer = model.get_input_embeddings()
    hidden_dim = 2560

    data = load_dataset()
    print(f"Loaded {len(data)} factual QA pairs from cache.")

    # Distractor set: 50 held-out facts strictly never ingested
    distractors = data[10000:10050]
    train_pool = data[:10000]

    # Milestone increments for scaling curve
    milestones = [100, 500, 1000, 2500, 5000, 10000]

    results = []

    # Initialize scalable memory
    memory = PopBrainScale10kMemory(
        hidden_dim=hidden_dim,
        num_clusters=256,
        nprobe=4,
        inhibition_threshold=0.85,
        margin_threshold=0.02,
        modulation_strength=0.18,
        device=device
    )

    currently_ingested = 0

    for N in milestones:
        print(f"\n" + "-" * 60)
        print(f"Target Memory Load: N = {N} facts")
        print("-" * 60)

        # Ingest delta facts in batches of 32 to guarantee headroom on RTX 3060
        t0_ingest = time.time()
        to_add = train_pool[currently_ingested:N]
        batch_size = 32

        for b_start in range(0, len(to_add), batch_size):
            b_chunk = to_add[b_start : b_start + batch_size]
            prompts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in b_chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
                hidden_states = out.hidden_states[-1] # [B, T, D]
                embs = embed_layer(enc.input_ids)    # [B, T, D]

                # Extract valid (K, V) transitions per sequence based on attention_mask
                chunk_keys = []
                chunk_vals = []
                for bi in range(len(b_chunk)):
                    seq_len = int(enc.attention_mask[bi].sum().item())
                    if seq_len >= 2:
                        h_seq = hidden_states[bi, :seq_len]
                        e_seq = embs[bi, :seq_len]
                        
                        k = h_seq[:-1]
                        k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
                        v = e_seq[1:]
                        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
                        
                        chunk_keys.append(k)
                        chunk_vals.append(v)

                del out, hidden_states, embs, enc
                if chunk_keys:
                    batch_k = torch.cat(chunk_keys, dim=0)
                    batch_v = torch.cat(chunk_vals, dim=0)
                    memory.ingest_batch(batch_k, batch_v)
                    del batch_k, batch_v, chunk_keys, chunk_vals

            if b_start % 512 == 0:
                torch.cuda.empty_cache()

        dt_ingest = time.time() - t0_ingest
        currently_ingested = N

        # Build / update IVF-256 Index
        t0_index = time.time()
        memory.build_ivf_index(iterations=3)
        dt_index = time.time() - t0_index

        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        print(f"Total Transitions: {memory.num_memories} | Ingest time: {dt_ingest:.2f}s | IVF build: {dt_index:.2f}s | VRAM: {vram_gb:.2f} GB")

        # 1. Evaluate Recall on 50 representative sampled stored queries
        eval_indices = torch.linspace(0, N - 1, steps=min(50, N)).long().tolist()
        recall_hits = 0
        latencies = []

        for idx in eval_indices:
            item = train_pool[idx]
            q = item["question"]
            target = item["answer"].strip().lower()
            prompt = f"Question: {q}\nAnswer:"
            curr_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]

            gen_tokens = []
            for _ in range(8):
                t_step0 = time.perf_counter()
                with torch.no_grad():
                    out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                    h_last = out.hidden_states[-1][0, -1, :]
                    delta_h, max_sim = memory.compute_synaptic_modulation(h_last)
                    if delta_h.norm() > 0:
                        h_mod = h_last + delta_h
                        logits = model.lm_head(h_mod.unsqueeze(0))[0]
                    else:
                        logits = out.logits[0, -1, :]
                    next_tok = torch.argmax(logits).unsqueeze(0)
                latencies.append((time.perf_counter() - t_step0) * 1000)

                curr_ids = torch.cat([curr_ids, next_tok])
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id or "\n" in tokenizer.decode(next_tok):
                    break

            pred = tokenizer.decode(gen_tokens).strip().lower()
            if target in pred:
                recall_hits += 1

        recall_rate = (recall_hits / len(eval_indices)) * 100.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # 2. Evaluate Distractor Interference on 50 unlearned queries
        interferences = 0
        for item in distractors:
            q = item["question"]
            prompt = f"Question: {q}\nAnswer:"
            curr_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
            with torch.no_grad():
                out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                h_last = out.hidden_states[-1][0, -1, :]
                delta_h, _ = memory.compute_synaptic_modulation(h_last)
                if delta_h.norm() > 0:
                    interferences += 1

        interference_rate = (interferences / len(distractors)) * 100.0

        print(f"Results for N = {N}:")
        print(f"  * Recall Rate:       {recall_rate:.1f}% ({recall_hits}/{len(eval_indices)})")
        print(f"  * Interference Rate: {interference_rate:.1f}% ({interferences}/{len(distractors)})")
        print(f"  * Mean Step Latency: {avg_latency:.2f} ms")

        results.append({
            "N": N,
            "transitions": memory.num_memories,
            "recall": recall_rate,
            "interference": interference_rate,
            "latency_ms": avg_latency,
            "vram_gb": vram_gb
        })

    # Save benchmark json results
    os.makedirs("assets/capacity_study", exist_ok=True)
    with open("assets/capacity_study/capacity_10k_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot Scientific 10k Scaling Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    ns = [r["N"] for r in results]
    recalls = [r["recall"] for r in results]
    interfs = [r["interference"] for r in results]
    vrams = [r["vram_gb"] for r in results]

    # Panel A: Recall & Interference vs N
    ax1.plot(ns, recalls, "o-", color="#1a73e8", linewidth=2.5, markersize=7, label="Recall Rate R(N) [%]")
    ax1.plot(ns, interfs, "s--", color="#d93025", linewidth=2.0, markersize=6, label="Distractor Cross-Talk I(N) [%]")
    ax1.set_xscale("log")
    ax1.set_xlabel("Stored Facts Load $N$ (Log Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Metric Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-2, 105)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_title("Associative Recall vs. Cross-Talk Interference (10,000 Facts)", fontsize=12, fontweight="bold")
    ax1.legend(loc="center right", frameon=True)

    # Panel B: Memory Footprint & Ingestion Scaling
    ax2.plot(ns, vrams, "d-", color="#137333", linewidth=2.5, markersize=7, label="Total VRAM Allocated (GB)")
    ax2.axhline(12.0, color="#80868b", linestyle=":", label="RTX 3060 Total VRAM Limit (12 GB)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Stored Facts Load $N$ (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("VRAM Usage (GB)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 13)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_title("Hardware Memory Scaling (RTX 3060 12GB)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plot_path = "assets/capacity_study/capacity_10k_scaling.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nScientific chart successfully saved to {plot_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
