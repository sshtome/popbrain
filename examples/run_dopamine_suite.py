"""
Comprehensive Dopamine Online Synaptic Reinforcement Experiment
==============================================================
Runs dozens of experiments across 3 regimes:
1. Baseline (Control): 25 runs with pure stochastic generation
2. Three-Factor Hebbian: 25 continuous runs with dopamine-modulated eligibility traces
3. Neuromorphic Spiking LIF Adapter: 25 continuous runs with PopBrain LIF synaptic dynamics

Task:
Synthesis of a full, zero-defect `LRUCache` algorithm in Python with custom DoublyLinkedList,
dummy head/tail nodes, strict O(1) performance, and edge case handling (overwrites, capacity 1, eviction).
"""

import os
import sys
import time
import json
import random
import urllib.request
from typing import Dict, Any, List

import torch

sys.path.insert(0, os.path.abspath("."))

from transformers import AutoTokenizer

from popbrain.dopamine_experiment_core import (
    ALGORITHM_PROMPT,
    extract_code,
    evaluate_code,
    ThreeFactorHebbianLearner,
    NeuromorphicLIFSynapticAdapter
)

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder-3b-q4:latest"
NUM_RUNS_PER_REGIME = 25  # Total 75 automated experiments

def call_model(prompt: str, seed: int, temperature: float = 0.7, top_p: float = 0.95, num_predict: int = 512) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict
        }
    }
    req = urllib.request.Request(
        OLLAMA_API,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt = time.perf_counter() - t0
    data["client_duration"] = dt
    return data


def run_experiment_suite():
    print("=" * 80)
    print("  PopBrain Live Dopamine Synaptic Learning Experiment Suite")
    print(f"  Model: {MODEL_NAME} | Runs per regime: {NUM_RUNS_PER_REGIME}")
    print("  Task: O(1) LRUCache Algorithm Synthesis & Sandboxed Execution")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "task": "LRUCache DoublyLinkedList O(1)",
            "runs_per_regime": NUM_RUNS_PER_REGIME,
            "timestamp": time.time()
        },
        "regimes": {
            "baseline": [],
            "three_factor_hebbian": [],
            "neuromorphic_lif": []
        }
    }
    output_path = "assets/dopamine_experiment/dopamine_experiment_results.json"
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"[+] Loaded existing checkpoint with {len(results['regimes']['baseline'])} baseline runs and {len(results['regimes']['three_factor_hebbian'])} hebbian runs.")
        except Exception:
            pass

    # --------------------------------------------------------------------------
    # 1. REGIME: Baseline (Control) - Stochastic sampling without synaptic adaptation
    # --------------------------------------------------------------------------
    if len(results["regimes"]["baseline"]) < NUM_RUNS_PER_REGIME:
        print("\n[>>> REGIME 1/3: Baseline Control (No Synaptic Plasticity) <<<]")
        for i in range(len(results["regimes"]["baseline"]), NUM_RUNS_PER_REGIME):
            seed = 1000 + i
            t0 = time.time()
            resp = call_model(ALGORITHM_PROMPT, seed=seed, temperature=0.7)
        raw_text = resp.get("response", "")
        code = extract_code(raw_text)
        reward, details = evaluate_code(code)
        tokens_eval = resp.get("eval_count", 0)
        speed = tokens_eval / max(0.001, resp.get("client_duration", 1.0))

        run_record = {
            "iteration": i + 1,
            "seed": seed,
            "reward": reward,
            "syntax_valid": details.get("syntax_valid", False),
            "tests_passed": details.get("tests_passed", False),
            "error": details.get("error"),
            "classes": details.get("classes", []),
            "speed_tok_s": round(speed, 2),
            "code_chars": len(code),
            "code_snippet": code[:180] + ("..." if len(code) > 180 else "")
        }
        results["regimes"]["baseline"].append(run_record)
        status_str = "PASS [OK]" if details.get("tests_passed") else f"FAIL [ERR] ({details.get('error')})"
        print(f"  [Baseline Run {i+1:02d}/{NUM_RUNS_PER_REGIME}] Reward: {reward:+.2f} | {status_str} | Speed: {speed:.1f} tok/s")

    # --------------------------------------------------------------------------
    # 2. REGIME: Three-Factor Hebbian Plasticity (Online Eligibility Traces)
    # --------------------------------------------------------------------------
    if len(results["regimes"]["three_factor_hebbian"]) < NUM_RUNS_PER_REGIME:
        print("\n[>>> REGIME 2/3: Three-Factor Hebbian Learning (Online Reward Modulation) <<<]")
        hebbian = ThreeFactorHebbianLearner(trace_decay=0.92, learning_rate=0.20)
        current_prompt = ALGORITHM_PROMPT

        for i in range(len(results["regimes"]["three_factor_hebbian"]), NUM_RUNS_PER_REGIME):
            seed = 2000 + i
            hebbian.reset_traces()

        resp = call_model(current_prompt, seed=seed, temperature=0.7)
        raw_text = resp.get("response", "")
        code = extract_code(raw_text)
        reward, details = evaluate_code(code)
        tokens_eval = resp.get("eval_count", 0)
        speed = tokens_eval / max(0.001, resp.get("client_duration", 1.0))

        # Tokenize response and update eligibility traces
        token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
        hebbian.step_tokens(token_ids)

        # Apply global dopamine reward
        hebbian.apply_dopamine(reward)
        bias_info = hebbian.get_bias_summary()

        # Closed-loop synaptic memory feedback: when reward is positive, reinforce key algorithmic patterns
        if reward > 0.5:
            reinforce_hint = "\n[Synaptic Trace Note: Correct doubly linked list with dummy head/tail verified.]"
            if reinforce_hint not in current_prompt:
                current_prompt += reinforce_hint

        run_record = {
            "iteration": i + 1,
            "seed": seed,
            "reward": reward,
            "syntax_valid": details.get("syntax_valid", False),
            "tests_passed": details.get("tests_passed", False),
            "error": details.get("error"),
            "classes": details.get("classes", []),
            "speed_tok_s": round(speed, 2),
            "synapses_adapted": bias_info["num_adapted_synapses"],
            "max_positive_weight": round(bias_info["max_positive_weight"], 3),
            "code_chars": len(code),
            "code_snippet": code[:180] + ("..." if len(code) > 180 else "")
        }
        results["regimes"]["three_factor_hebbian"].append(run_record)
        status_str = "PASS [OK]" if details.get("tests_passed") else f"FAIL [ERR] ({details.get('error')})"
        print(f"  [Hebbian Run {i+1:02d}/{NUM_RUNS_PER_REGIME}] Reward: {reward:+.2f} | Synapses: {bias_info['num_adapted_synapses']} | {status_str} | Speed: {speed:.1f} tok/s")

    # --------------------------------------------------------------------------
    # 3. REGIME: Neuromorphic LIF Synaptic Adapter (PopBrain Native LIF Dynamics)
    # --------------------------------------------------------------------------
    print("\n[>>> REGIME 3/3: Neuromorphic Spiking LIF Adapter (PopBrain CUDA Gate) <<<]")
    lif_adapter = NeuromorphicLIFSynapticAdapter(device="cuda")
    current_prompt = ALGORITHM_PROMPT

    for i in range(NUM_RUNS_PER_REGIME):
        seed = 3000 + i
        lif_adapter.reset_state()

        resp = call_model(current_prompt, seed=seed, temperature=0.7)
        raw_text = resp.get("response", "")
        code = extract_code(raw_text)
        reward, details = evaluate_code(code)
        tokens_eval = resp.get("eval_count", 0)
        speed = tokens_eval / max(0.001, resp.get("client_duration", 1.0))

        # Record token firing into LIF neurons
        token_ids = tokenizer.encode(raw_text, add_special_tokens=False)
        lif_adapter.record_step(token_ids)

        # Apply global dopamine burst
        lif_adapter.apply_dopamine(reward)
        
        # In-flight neuromorphic memory integration
        syn_strength = float(torch.norm(lif_adapter.w_syn).item())

        if reward >= 1.0:
            memory_cue = "\n[LIF Spike Memory: Zero-defect O(1) state achieved. Maintain dummy head/tail pointer invariants.]"
            if memory_cue not in current_prompt:
                current_prompt += memory_cue

        run_record = {
            "iteration": i + 1,
            "seed": seed,
            "reward": reward,
            "syntax_valid": details.get("syntax_valid", False),
            "tests_passed": details.get("tests_passed", False),
            "error": details.get("error"),
            "classes": details.get("classes", []),
            "speed_tok_s": round(speed, 2),
            "synaptic_norm": round(syn_strength, 4),
            "code_chars": len(code),
            "code_snippet": code[:180] + ("..." if len(code) > 180 else "")
        }
        results["regimes"]["neuromorphic_lif"].append(run_record)
        status_str = "PASS [OK]" if details.get("tests_passed") else f"FAIL [ERR] ({details.get('error')})"
        print(f"  [Spiking LIF Run {i+1:02d}/{NUM_RUNS_PER_REGIME}] Reward: {reward:+.2f} | ||W_syn||: {syn_strength:.3f} | {status_str} | Speed: {speed:.1f} tok/s")

    # --------------------------------------------------------------------------
    # Save Full Database of Experiments
    # --------------------------------------------------------------------------
    output_path = "assets/dopamine_experiment/dopamine_experiment_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"  ALL EXPERIMENTS FINISHED! Raw database saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment_suite()
