"""
Scientific Benchmark: Memory Capacity vs. Synaptic Cross-Talk Interference
==========================================================================
Evaluates the scaling limits of PopBrain Hetero-Associative Synaptic Memory
with GABAergic Lateral Inhibition on SQuAD factual question-answer pairs
using Microsoft BitNet-b1.58-2B-4T (2.4B parameters).

Plots:
------
1. Factual Recall Rate (%) vs. Memory Load (N)
2. Cross-Talk / Interference Rate (%) on unlearned distractor questions
3. Control Domain General Retention (%) (Catastrophic Forgetting resistance)
4. O(1) Ingestion Latency (ms / fact)

Usage:
------
python examples/run_capacity_vs_interference_study.py
"""

import os
import sys
import time
import json
import argparse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from popbrain import SynapticCapacityBenchmark


def main():
    parser = argparse.ArgumentParser(description="PopBrain Capacity vs. Interference Benchmark")
    parser.add_argument("--model-path", type=str, default="D:/projects/models/bitnet-b1.58-2B-4T")
    parser.add_argument("--milestones", type=str, default="5,10,20,30,40,50")
    parser.add_argument("--distractors", type=int, default=15)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.18)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    milestones = [int(x.strip()) for x in args.milestones.split(",")]

    print("=" * 85)
    print("  PopBrain Benchmark: Synaptic Memory Capacity vs. Cross-Talk Interference")
    print(f"  Model: BitNet-b1.58-2B-4T (2.4B Params) | Device: {device.upper()}")
    print(f"  Plasticity: Hetero-Associative SNN | Lateral Inhibition Threshold: {args.threshold}")
    print("=" * 85)

    # 1. Dataset Preparation
    print("\n[Phase 1] Preparing SQuAD Factual Dataset...")
    dataset = SynapticCapacityBenchmark.ensure_squad_dataset(
        cache_path="assets/datasets/squad_sampled.json",
        max_items=max(milestones) + args.distractors + 20
    )
    print(f"[+] Loaded {len(dataset)} factual QA pairs from SQuAD dataset cache.")

    # 2. Model Initialization
    print("\n[Phase 2] Loading Model into VRAM...")
    tok = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()

    # 3. Benchmark Harness
    benchmark = SynapticCapacityBenchmark(
        model=model,
        tokenizer=tok,
        hidden_dim=model.config.hidden_size,
        inhibition_threshold=args.threshold,
        modulation_strength=args.alpha,
        device=device
    )

    # 4. Execute Scaling Run
    print(f"\n[Phase 3] Running Capacity Scaling across Milestones: {milestones}...")
    print(f"  Evaluating recall on stored items and cross-talk on {args.distractors} unlearned distractor questions.\n")

    t_start = time.time()
    results = benchmark.run_capacity_experiment(
        qa_dataset=dataset,
        milestones=milestones,
        num_distractors=args.distractors
    )
    total_time = time.time() - t_start

    print(f"\n[+] Capacity benchmark completed in {total_time:.2f} seconds.")

    # 5. Generate Publication-Quality Scientific Plot
    print("\n[Phase 4] Generating Scientific Figures...")
    os.makedirs("assets/capacity_study", exist_ok=True)
    plot_path = "assets/capacity_study/capacity_vs_interference.png"
    benchmark.plot_capacity_vs_interference(results, save_path=plot_path)

    # 6. Save JSON Data
    json_path = "assets/capacity_study/capacity_vs_interference_results.json"
    results["metadata"] = {
        "model": "BitNet-b1.58-2B-4T",
        "parameters": "2.4B",
        "device": device,
        "threshold": args.threshold,
        "alpha": args.alpha,
        "total_runtime_sec": round(total_time, 2)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved metrics JSON to: {json_path}")

    # 7. Print Executive Summary Table
    print("\n" + "=" * 85)
    print("  EXECUTIVE SUMMARY TABLE:")
    print("  Memory Load (N) | Recall Rate (%) | Interference (%) | Control Retain (%) | Latency (ms/fact)")
    print("  " + "-" * 75)
    for i, n in enumerate(results["milestones"]):
        rec = results["recall_rates"][i]
        interf = results["interference_rates"][i]
        ctrl = results["control_retention"][i]
        lat = results["ingestion_latencies_ms"][i]
        print(f"       N = {n:2d}      |     {rec:5.1f}%     |      {interf:5.1f}%     |      {ctrl:5.1f}%       |     {lat:4.1f} ms")
    print("=" * 85)


if __name__ == "__main__":
    main()
