"""
Comparative Scientific Benchmark: Hierarchical Trajectory Memory vs. Flat Synaptic Memory
==========================================================================================
Empirical scaling evaluation comparing:
1. Baseline Flat Synaptic Memory (Single-state thresholding)
2. PopBrain Hierarchical Synaptic Memory:
   - Multi-Scale Contextual Trajectory Binding (W=3 step temporal binding)
   - Hierarchical GPU Cluster Gating (Spherical k-means Voronoi routing)
   - Differential Margin Competition (Top-1 vs Top-2 contrastive inhibition)

Tested on 100 SQuAD QA pairs using Microsoft BitNet-b1.58-2B-4T on NVIDIA RTX 3060.
"""

import os
import sys
import time
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from popbrain import PopBrainSynapticMemory, PopBrainHierarchicalSynapticMemory
from popbrain.dopamine.capacity_evaluator import SynapticCapacityBenchmark


def run_comparative_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "D:/projects/models/bitnet-b1.58-2B-4T"
    milestones = [10, 25, 50, 75, 100]
    num_distractors = 20

    print("=" * 85)
    print("  PopBrain Hierarchical vs. Flat Synaptic Memory Scaling Benchmark")
    print(f"  Architecture: BitNet-b1.58-2B-4T (2.4B Params) | Device: {device.upper()}")
    print("  Methodologies: 1. Trajectory Binding | 2. Margin Competition | 3. Cluster Gating")
    print("=" * 85)

    # 1. Dataset
    print("\n[Phase 1] Loading SQuAD Factual Dataset...")
    dataset = SynapticCapacityBenchmark.ensure_squad_dataset(
        cache_path="assets/datasets/squad_sampled.json",
        max_items=150
    )
    train_pool = dataset[:100]
    distractor_pool = dataset[100 : 100 + num_distractors]
    print(f"[+] Loaded {len(train_pool)} training facts and {len(distractor_pool)} distractor questions.")

    # 2. Model
    print("\n[Phase 2] Loading Model into VRAM...")
    tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()

    embed_layer = model.get_input_embeddings()
    hidden_dim = model.config.hidden_size

    # 3. Initialize Both Architectures
    flat_mem = PopBrainSynapticMemory(
        hidden_dim=hidden_dim,
        inhibition_threshold=0.85,
        modulation_strength=0.18,
        device=device
    )

    hier_mem = PopBrainHierarchicalSynapticMemory(
        hidden_dim=hidden_dim,
        window_size=3,
        gamma=0.5,
        inhibition_threshold=0.80,
        margin_threshold=0.03,
        modulation_strength=0.18,
        num_clusters=16,
        device=device
    )

    results = {
        "milestones": milestones,
        "flat": {"recall": [], "interference": [], "search_ms": []},
        "hierarchical": {"recall": [], "interference": [], "search_ms": []}
    }

    # Helper generation
    def evaluate_model(mem_obj, is_hierarchical: bool, current_n: int) -> Tuple[float, float, float]:
        # 1. Recall
        correct = 0
        search_times = []
        for item in train_pool[:current_n]:
            if is_hierarchical:
                mem_obj.reset_runtime_trajectory()
            prompt = f"Question: {item['question']}\nAnswer:"
            curr_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
            gen_tokens = []
            
            for _ in range(8):
                with torch.no_grad():
                    out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                    h_last = out.hidden_states[-1][0, -1, :]

                    t_s0 = time.time()
                    if is_hierarchical:
                        h_seq_ctx = list(out.hidden_states[-1][0, -mem_obj.window_size:])
                        delta_h, _ = mem_obj.compute_synaptic_modulation(h_last, context_history=h_seq_ctx)
                    else:
                        delta_h, _ = mem_obj.compute_synaptic_modulation(h_last)
                    search_times.append((time.time() - t_s0) * 1000)

                    if delta_h.norm() > 0:
                        logits = model.lm_head((h_last + delta_h).unsqueeze(0))[0]
                    else:
                        logits = out.logits[0, -1, :]

                    nxt = int(torch.argmax(logits).item())
                    curr_ids = torch.cat([curr_ids, torch.tensor([nxt], device=device)])
                    gen_tokens.append(nxt)
                    if nxt == tok.eos_token_id or "\n" in tok.decode([nxt]):
                        break

            ans = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            if item["answer"].lower() in ans.lower():
                correct += 1

        recall = (correct / current_n) * 100.0

        # 2. Interference on 20 distractor questions
        false_positives = 0
        for dist in distractor_pool:
            if is_hierarchical:
                mem_obj.reset_runtime_trajectory()
            prompt = f"Question: {dist['question']}\nAnswer:"
            curr_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
            with torch.no_grad():
                out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                h_last = out.hidden_states[-1][0, -1, :]
                if is_hierarchical:
                    h_seq_ctx = list(out.hidden_states[-1][0, -mem_obj.window_size:])
                    delta_h, max_sim = mem_obj.compute_synaptic_modulation(h_last, context_history=h_seq_ctx)
                else:
                    delta_h, max_sim = mem_obj.compute_synaptic_modulation(h_last)
                if delta_h.norm() > 0:
                    false_positives += 1

        interference = (false_positives / len(distractor_pool)) * 100.0
        avg_search_ms = sum(search_times) / max(1, len(search_times))
        return recall, interference, avg_search_ms

    # 4. Progressive Ingestion and Evaluation
    print("\n[Phase 3] Running Comparative Scaling Study...")
    prev_n = 0

    for target_n in milestones:
        items_to_add = train_pool[prev_n:target_n]

        # Ingest into both architectures
        for item in items_to_add:
            full_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            ids = tok(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
            with torch.no_grad():
                out = model(ids.unsqueeze(0), output_hidden_states=True)
                h_seq = out.hidden_states[-1][0]
                e_seq = embed_layer(ids)

                flat_mem.ingest_sequence(h_seq, e_seq, reward=1.0)
                hier_mem.ingest_sequence(h_seq, e_seq, reward=1.0)

        prev_n = target_n

        # Evaluate Flat
        flat_rec, flat_int, flat_ms = evaluate_model(flat_mem, is_hierarchical=False, current_n=target_n)
        results["flat"]["recall"].append(round(flat_rec, 2))
        results["flat"]["interference"].append(round(flat_int, 2))
        results["flat"]["search_ms"].append(round(flat_ms, 3))

        # Evaluate Hierarchical
        hier_rec, hier_int, hier_ms = evaluate_model(hier_mem, is_hierarchical=True, current_n=target_n)
        results["hierarchical"]["recall"].append(round(hier_rec, 2))
        results["hierarchical"]["interference"].append(round(hier_int, 2))
        results["hierarchical"]["search_ms"].append(round(hier_ms, 3))

        print(f"  [Load N={target_n:03d} Facts | {hier_mem.num_memories:4d} Transitions]")
        print(f"    - Flat Baseline Memory  : Recall = {flat_rec:5.1f}% | Interference = {flat_int:4.1f}% | Search = {flat_ms:4.2f} ms")
        print(f"    - Hierarchical PopBrain : Recall = {hier_rec:5.1f}% | Interference = {hier_int:4.1f}% | Search = {hier_ms:4.2f} ms")

    # 5. Plot Publication Comparison
    print("\n[Phase 4] Generating Comparative Scientific Figures...")
    os.makedirs("assets/capacity_study", exist_ok=True)
    plot_path = "assets/capacity_study/hierarchical_vs_flat_scaling.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel A: Recall Rate
    ax1.plot(milestones, results["hierarchical"]["recall"], marker="o", color="#1f77b4", linewidth=2.5, label="PopBrain Hierarchical (Trajectory+Margin)")
    ax1.plot(milestones, results["flat"]["recall"], marker="s", color="#7f7f7f", linestyle="--", linewidth=2.0, label="Flat Synaptic Memory (Single-state)")
    ax1.set_title("A. Factual Recall Rate vs. Memory Load (N=100)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Stored Facts (N)", fontsize=10)
    ax1.set_ylabel("Factual Recall Rate (%)", fontsize=10)
    ax1.set_ylim(40, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)

    # Panel B: Cross-Talk Interference
    ax2.plot(milestones, results["hierarchical"]["interference"], marker="D", color="#2ca02c", linewidth=2.5, label="PopBrain Hierarchical Interference (%)")
    ax2.plot(milestones, results["flat"]["interference"], marker="^", color="#d62728", linestyle="--", linewidth=2.0, label="Flat Memory Interference (%)")
    ax2.set_title("B. Cross-Talk Interference Suppression", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Stored Facts (N)", fontsize=10)
    ax2.set_ylabel("False Positive Activation Rate (%)", fontsize=10)
    ax2.set_ylim(-2, 35)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparative chart to: {plot_path}")

    # 6. Save JSON Data
    json_path = "assets/capacity_study/hierarchical_vs_flat_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved comparative JSON metrics to: {json_path}")

    # Summary table
    print("\n" + "=" * 85)
    print("  COMPARATIVE SUMMARY (N=100 FACTS):")
    print(f"  - Flat Baseline Recall (N=100)        : {results['flat']['recall'][-1]:.1f}%")
    print(f"  - Hierarchical PopBrain Recall (N=100): {results['hierarchical']['recall'][-1]:.1f}%")
    print(f"  - Flat Interference Rate (N=100)      : {results['flat']['interference'][-1]:.1f}%")
    print(f"  - Hierarchical Interference (N=100)   : {results['hierarchical']['interference'][-1]:.1f}%")
    print("=" * 85)


if __name__ == "__main__":
    run_comparative_experiment()
