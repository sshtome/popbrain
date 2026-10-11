"""
PopBrain Synaptic Memory Capacity & Interference Evaluation Engine
==================================================================
Empirical evaluation harness for measuring associative memory capacity,
cross-talk interference, and control retention scaling as a function of
stored memories (N) in frozen Large Language Models without backpropagation.

Mathematical Formulations:
--------------------------
1. Memory Load (N):
   Number of distinct factual associations ingested into synaptic memory:
       M_N = {(K_i, V_i)}_{i=1}^N

2. Factual Recall Rate R(N):
   Fraction of stored queries whose greedy generation reproduces target answer:
       R(N) = (1/N) * sum_{i=1}^N [ Target_i in Gen(Q_i | M_N) ]

3. Cross-Talk / Interference Rate I(N):
   Spurious activation rate on held-out unlearned distractor queries:
       I(N) = (1/|D|) * sum_{j in D} [ max_k cos(K_k, h(Q_j)) >= theta_inhib ]

4. Ingestion Latency T_ingest(N):
   Forward-pass processing time per fact (strictly O(1) in SNN / single-pass memory).
"""

from __future__ import annotations

import os
import sys
import time
import json
import urllib.request
from typing import List, Dict, Tuple, Any, Optional

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from popbrain.dopamine.synaptic_memory import PopBrainSynapticMemory


class SynapticCapacityBenchmark:
    """
    Automated evaluation harness for Memory Capacity vs. Interference.
    """
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        hidden_dim: int = 2560,
        inhibition_threshold: float = 0.85,
        modulation_strength: float = 0.18,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.inhibition_threshold = inhibition_threshold
        self.modulation_strength = modulation_strength
        self.device = device
        self.embed_layer = model.get_input_embeddings()

    @staticmethod
    def ensure_squad_dataset(cache_path: str = "assets/datasets/squad_sampled.json", max_items: int = 100) -> List[Dict[str, str]]:
        """Downloads and caches a clean sample of factual QA pairs from SQuAD."""
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)

        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        qa_pairs = []
        for topic in data["data"]:
            for p in topic["paragraphs"]:
                for qa in p["qas"]:
                    if not qa.get("is_impossible", False) and qa.get("answers"):
                        ans = qa["answers"][0]["text"].strip()
                        q = qa["question"].strip()
                        if 1 <= len(ans.split()) <= 6 and len(ans) > 2:
                            qa_pairs.append({"question": q, "answer": ans})
                    if len(qa_pairs) >= max_items:
                        break
                if len(qa_pairs) >= max_items: break
            if len(qa_pairs) >= max_items: break

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
        return qa_pairs

    def generate_answer(
        self,
        question: str,
        memory: PopBrainSynapticMemory,
        max_tokens: int = 10
    ) -> Tuple[str, float]:
        """Generates answer for query and returns (generated_text, max_observed_similarity)."""
        prompt = f"Question: {question}\nAnswer:"
        curr_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)[0]
        gen_tokens = []
        highest_sim = 0.0

        for _ in range(max_tokens):
            with torch.no_grad():
                out = self.model(curr_ids.unsqueeze(0), output_hidden_states=True)
                h_last = out.hidden_states[-1][0, -1, :]

                delta_h, max_sim = memory.compute_synaptic_modulation(h_last)
                highest_sim = max(highest_sim, max_sim)

                if delta_h.norm() > 0:
                    h_mod = h_last + delta_h
                    logits = self.model.lm_head(h_mod.unsqueeze(0))[0]
                else:
                    logits = out.logits[0, -1, :]

                nxt = int(torch.argmax(logits).item())
                curr_ids = torch.cat([curr_ids, torch.tensor([nxt], device=self.device)])
                gen_tokens.append(nxt)

                if nxt == self.tokenizer.eos_token_id or "\n" in self.tokenizer.decode([nxt]):
                    break

        ans = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return ans, highest_sim

    def run_capacity_experiment(
        self,
        qa_dataset: List[Dict[str, str]],
        milestones: List[int] = [5, 10, 20, 30, 40, 50],
        num_distractors: int = 15,
        control_prompts: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Runs progressive memory load scaling and measures recall, interference, and latency.
        """
        if control_prompts is None:
            control_prompts = [
                {"prompt": "The capital of France is", "target": "Paris"},
                {"prompt": "Python is a programming language known for its", "target": "readability"},
                {"prompt": "Water boils at a temperature of 100 degrees", "target": "Celsius"}
            ]

        max_n = max(milestones)
        train_pool = qa_dataset[:max_n]
        distractor_pool = qa_dataset[max_n : max_n + num_distractors]

        results = {
            "milestones": milestones,
            "recall_rates": [],
            "interference_rates": [],
            "control_retention": [],
            "ingestion_latencies_ms": [],
            "cumulative_memories": []
        }

        memory = PopBrainSynapticMemory(
            hidden_dim=self.hidden_dim,
            inhibition_threshold=self.inhibition_threshold,
            modulation_strength=self.modulation_strength,
            device=self.device
        )

        curr_ingested = 0

        for target_n in milestones:
            items_to_add = train_pool[curr_ingested:target_n]
            t0 = time.time()

            # Progressive ingestion
            for item in items_to_add:
                full_text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                ids = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)[0]
                with torch.no_grad():
                    out = self.model(ids.unsqueeze(0), output_hidden_states=True)
                    h_seq = out.hidden_states[-1][0]
                    e_seq = self.embed_layer(ids)
                    memory.ingest_sequence(h_seq, e_seq, reward=1.0)

            ingest_time_ms = (time.time() - t0) * 1000 / max(1, len(items_to_add))
            curr_ingested = target_n

            # 1. Measure Recall on all target_n stored items
            correct_recall = 0
            for item in train_pool[:target_n]:
                ans, _ = self.generate_answer(item["question"], memory)
                if item["answer"].lower() in ans.lower():
                    correct_recall += 1
            recall_rate = (correct_recall / target_n) * 100.0

            # 2. Measure Interference on unlearned distractors
            false_positives = 0
            for dist in distractor_pool:
                _, sim = self.generate_answer(dist["question"], memory)
                if sim >= self.inhibition_threshold:
                    false_positives += 1
            interference_rate = (false_positives / len(distractor_pool)) * 100.0

            # 3. Measure Control Domain Retention
            ctrl_pass = 0
            for cp in control_prompts:
                ids = self.tokenizer(cp["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)[0]
                with torch.no_grad():
                    out = self.model(ids.unsqueeze(0), output_hidden_states=True)
                    h_last = out.hidden_states[-1][0, -1, :]
                    delta_h, _ = memory.compute_synaptic_modulation(h_last)
                    logits = self.model.lm_head((h_last + delta_h).unsqueeze(0))[0]
                    next_tok = self.tokenizer.decode([int(torch.argmax(logits).item())])
                    if cp["target"].lower() in next_tok.lower():
                        ctrl_pass += 1
            control_score = (ctrl_pass / len(control_prompts)) * 100.0

            results["recall_rates"].append(round(recall_rate, 2))
            results["interference_rates"].append(round(interference_rate, 2))
            results["control_retention"].append(round(control_score, 2))
            results["ingestion_latencies_ms"].append(round(ingest_time_ms, 2))
            results["cumulative_memories"].append(memory.num_memories)

            print(f"  [Load N={target_n:02d}] Recall: {recall_rate:5.1f}% | Interference: {interference_rate:4.1f}% | "
                  f"Control Retain: {control_score:5.1f}% | Transitions: {memory.num_memories:4d} | "
                  f"Ingest: {ingest_time_ms:4.1f} ms/fact")

        return results

    @staticmethod
    def plot_capacity_vs_interference(results: Dict[str, Any], save_path: str = "assets/capacity_study/capacity_vs_interference.png"):
        """Generates a publication-quality scientific chart."""
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        milestones = results["milestones"]
        recall = results["recall_rates"]
        interference = results["interference_rates"]
        control = results["control_retention"]
        latencies = results["ingestion_latencies_ms"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

        # ---------------- Panel A: Capacity vs. Interference ----------------
        ax1.plot(milestones, recall, marker="o", color="#1f77b4", linewidth=2.5, markersize=7, label="Factual Recall Rate (Target Recall %)")
        ax1.plot(milestones, interference, marker="s", color="#d62728", linewidth=2.2, markersize=6, linestyle="--", label="Interference / Cross-Talk Rate (%)")
        ax1.plot(milestones, control, marker="^", color="#2ca02c", linewidth=2.0, markersize=6, linestyle=":", label="Control Domain Retention (No Forgetting)")

        ax1.set_title("A. Synaptic Memory Capacity vs. Cross-Talk Interference", fontsize=12, fontweight="bold", pad=12)
        ax1.set_xlabel("Number of Stored Facts (Memory Load N)", fontsize=11, labelpad=8)
        ax1.set_ylabel("Accuracy / Activation Rate (%)", fontsize=11, labelpad=8)
        ax1.set_ylim(-5, 105)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="center right", fontsize=9, framealpha=0.9)

        # ---------------- Panel B: Ingestion Latency (O(1) Scaling) ----------------
        ax2.plot(milestones, latencies, marker="D", color="#9467bd", linewidth=2.2, markersize=6, label="Ingestion Latency per Fact (ms)")
        ax2.axhline(y=sum(latencies)/len(latencies), color="#8c564b", linestyle="--", alpha=0.7, label=f"Mean Latency ({sum(latencies)/len(latencies):.1f} ms)")

        ax2.set_title("B. Single-Pass Ingestion Scalability (Zero-Backprop)", fontsize=12, fontweight="bold", pad=12)
        ax2.set_xlabel("Number of Stored Facts (Memory Load N)", fontsize=11, labelpad=8)
        ax2.set_ylabel("Ingestion Latency per Fact (ms)", fontsize=11, labelpad=8)
        ax2.set_ylim(0, max(latencies) * 1.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved scientific figure to: {save_path}")
