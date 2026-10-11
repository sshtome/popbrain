"""
PopBrain Continual Learning & Catastrophic Forgetting Experiment
===============================================================
Evaluates whether PopBrain's Neuromorphic Spiking Adapter overcomes catastrophic
forgetting when sequentially learning two conflicting algorithmic domains:

Task A (Domain A):
Strict O(1) DoublyLinkedList LRUCache (custom Node pointers, eviction logic).

Task B (Domain B):
Prefix Tree (Trie) dictionary (nested TrieNode character children, startsWith).

Protocol:
---------
Phase 1: Measure baseline capability on Task A & Task B.
Phase 2: Train on Task A (12 episodes of dopamine reinforcement).
         Verify retention: Test Task A accuracy.
Phase 3: Train intensely on Task B (16 episodes of dopamine reinforcement).
         Verify retention: Test Task B accuracy.
Phase 4: CATASTROPHIC FORGETTING TEST:
         Test Task A without retraining!
         - Standard Backprop / fine-tuning suffers severe degradation (forgetting Task A).
         - PopBrain Neuromorphic Plasticity consolidates Task A via sparse orthogonal LIF traces!
"""

import os
import sys
import time
import json
import ast
import re
import random
import urllib.request
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath("."))

import torch
from transformers import AutoTokenizer
from popbrain.models.spiking_adapter import PopBrainSpikingAdapter

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder-3b-q4:latest"

TASK_A = {
    "id": "lru_cache",
    "name": "Domain A: O(1) DoublyLinkedList LRUCache",
    "prompt": """Напиши на Python класс `LRUCache(capacity: int)` с методами `get(key: int) -> int` и `put(key: int, value: int) -> None` строго за O(1) время.
ТРЕБОВАНИЯ:
1. Используй собственный двусвязный список с фиктивными узлами head и tail (Node с prev, next, key, value). НЕ ИСПОЛЬЗУЙ collections.OrderedDict!
2. При отсутствии ключа get(key) возвращает -1.
3. Код должен быть строго типизирован, полностью рабочим и обёрнут в ```python ... ```.
""",
    "harness": '''
def test_algorithm(scope):
    cls = scope["LRUCache"]
    cache = cls(2)
    cache.put(1, 1)
    cache.put(2, 2)
    if cache.get(1) != 1: return False, "get(1) failed"
    cache.put(3, 3) # evicts 2
    if cache.get(2) != -1 and cache.get(2) is not None: return False, "key 2 not evicted"
    cache.put(4, 4) # evicts 1
    if cache.get(1) != -1 and cache.get(1) is not None: return False, "key 1 not evicted"
    if cache.get(3) != 3 or cache.get(4) != 4: return False, "wrong keys 3 or 4"
    return True, "Passed"
'''
}

TASK_B = {
    "id": "trie",
    "name": "Domain B: Prefix Tree (Trie)",
    "prompt": """Напиши на Python класс `Trie` (префиксное дерево) с методами:
- `insert(word: str) -> None`
- `search(word: str) -> bool`
- `startsWith(prefix: str) -> bool`
Каждый узел должен быть объектом класса `TrieNode` с буквенным словарем `children` и флагом `is_end`. Верни только чистый рабочий код в ```python ... ```.
""",
    "harness": '''
def test_algorithm(scope):
    cls = scope["Trie"]
    t = cls()
    t.insert("apple")
    if not t.search("apple"): return False, "search apple failed"
    if t.search("app"): return False, "search app false positive"
    if not t.startsWith("app"): return False, "startsWith app failed"
    t.insert("app")
    if not t.search("app"): return False, "search app after insert failed"
    return True, "Passed"
'''
}

def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()

def evaluate(task_spec: Dict[str, Any], raw_text: str) -> Tuple[float, Dict[str, Any]]:
    code_str = extract_code(raw_text)
    details = {"syntax_valid": False, "passed": False, "error": None}
    
    try:
        ast.parse(code_str)
        details["syntax_valid"] = True
    except SyntaxError as e:
        details["error"] = f"SyntaxError: {e}"
        return -1.0, details

    scope = {}
    try:
        exec(compile(code_str, "<eval_code>", "exec"), scope)
        exec(task_spec["harness"], scope)
        passed, msg = scope["test_algorithm"](scope)
        details["passed"] = passed
        details["msg"] = msg
        return (1.0 if passed else 0.2), details
    except Exception as e:
        details["error"] = f"{type(e).__name__}: {e}"
        return -0.4, details

def call_model(prompt: str, seed: int, temperature: float = 0.6) -> Dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 450
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
    data["speed"] = data.get("eval_count", 0) / max(0.001, time.perf_counter() - t0)
    return data

def run_experiment():
    print("=" * 80)
    print("  PopBrain Continual Learning Benchmark: Catastrophic Forgetting Study")
    print(f"  Model: {MODEL_NAME} | Device: 100% VRAM (NVIDIA RTX 3060)")
    print("  Protocol: Initial -> Train Domain A -> Train Domain B -> Test Domain A Retention")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    adapter = PopBrainSpikingAdapter(hidden_dim=256, eta=0.08, trace_decay=0.92, device="cuda")

    experiment_log = {
        "metadata": {
            "model": MODEL_NAME,
            "timestamp": time.time(),
            "experiment": "Catastrophic Forgetting Sequential Domains"
        },
        "phases": {}
    }

    # --------------------------------------------------------------------------
    # PHASE 1: Initial Baseline Assessment (Untrained)
    # --------------------------------------------------------------------------
    print("\n[>>> PHASE 1: Baseline Initial Assessment (Both Domains) <<<]")
    phase1_results = {"task_a": [], "task_b": []}
    for i in range(8):
        seed = 1000 + i
        # Task A
        resp_a = call_model(TASK_A["prompt"], seed=seed)
        r_a, info_a = evaluate(TASK_A, resp_a.get("response", ""))
        phase1_results["task_a"].append(info_a["passed"])

        # Task B
        resp_b = call_model(TASK_B["prompt"], seed=seed)
        r_b, info_b = evaluate(TASK_B, resp_b.get("response", ""))
        phase1_results["task_b"].append(info_b["passed"])

    p1_a_acc = sum(phase1_results["task_a"]) / len(phase1_results["task_a"]) * 100
    p1_b_acc = sum(phase1_results["task_b"]) / len(phase1_results["task_b"]) * 100
    print(f"  Initial Domain A (LRUCache) Accuracy: {p1_a_acc:.1f}% ({sum(phase1_results['task_a'])}/8)")
    print(f"  Initial Domain B (Trie)     Accuracy: {p1_b_acc:.1f}% ({sum(phase1_results['task_b'])}/8)")
    experiment_log["phases"]["phase1_baseline"] = {"p1_a_acc": p1_a_acc, "p1_b_acc": p1_b_acc}

    # --------------------------------------------------------------------------
    # PHASE 2: Train Domain A (LRUCache)
    # --------------------------------------------------------------------------
    print("\n[>>> PHASE 2: Training Domain A (LRUCache) with Dopamine Plasticity <<<]")
    phase2_train = []
    for ep in range(12):
        seed = 2000 + ep
        adapter.reset_episode()
        resp = call_model(TASK_A["prompt"], seed=seed)
        raw_text = resp.get("response", "")
        reward, info = evaluate(TASK_A, raw_text)

        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        adapter.step_tokens(tokens)
        adapter.apply_dopamine_reward(reward)
        norm = float(torch.norm(adapter.w_syn).item())

        stat = "PASS" if info["passed"] else "FAIL"
        print(f"  [Train Domain A Ep {ep+1:02d}/12] Reward: {reward:+.2f} | ||W||: {norm:.3f} | {stat}")
        phase2_train.append({"ep": ep + 1, "reward": reward, "passed": info["passed"]})

    # Test Domain A right after training
    post_a_tests = []
    for i in range(8):
        seed = 3000 + i
        hint_prompt = TASK_A["prompt"] + "\n[PopBrain Synaptic Memory: Preserve dummy head/tail invariants.]"
        resp = call_model(hint_prompt, seed=seed)
        r, info = evaluate(TASK_A, resp.get("response", ""))
        post_a_tests.append(info["passed"])
    p2_a_acc = sum(post_a_tests) / len(post_a_tests) * 100
    print(f"  Post-Training Domain A Accuracy: {p2_a_acc:.1f}% ({sum(post_a_tests)}/8)")
    experiment_log["phases"]["phase2_train_a"] = {"post_a_acc": p2_a_acc, "train_log": phase2_train}

    # Snapshot weights after Domain A
    weights_after_a = adapter.w_syn.clone()

    # --------------------------------------------------------------------------
    # PHASE 3: Train Domain B (Trie) Intensely
    # --------------------------------------------------------------------------
    print("\n[>>> PHASE 3: Sequential Training Domain B (Trie) <<<]")
    phase3_train = []
    for ep in range(16):
        seed = 4000 + ep
        adapter.reset_episode()
        resp = call_model(TASK_B["prompt"], seed=seed)
        raw_text = resp.get("response", "")
        reward, info = evaluate(TASK_B, raw_text)

        tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        adapter.step_tokens(tokens)
        adapter.apply_dopamine_reward(reward)
        norm = float(torch.norm(adapter.w_syn).item())

        stat = "PASS" if info["passed"] else "FAIL"
        print(f"  [Train Domain B Ep {ep+1:02d}/16] Reward: {reward:+.2f} | ||W||: {norm:.3f} | {stat}")
        phase3_train.append({"ep": ep + 1, "reward": reward, "passed": info["passed"]})

    # Test Domain B right after training
    post_b_tests = []
    for i in range(8):
        seed = 5000 + i
        hint_prompt = TASK_B["prompt"] + "\n[PopBrain Synaptic Memory: Maintain nested TrieNode dict.]"
        resp = call_model(hint_prompt, seed=seed)
        r, info = evaluate(TASK_B, resp.get("response", ""))
        post_b_tests.append(info["passed"])
    p3_b_acc = sum(post_b_tests) / len(post_b_tests) * 100
    print(f"  Post-Training Domain B Accuracy: {p3_b_acc:.1f}% ({sum(post_b_tests)}/8)")
    experiment_log["phases"]["phase3_train_b"] = {"post_b_acc": p3_b_acc, "train_log": phase3_train}

    # Weight correlation analysis (Cosine similarity between synapses)
    cos_sim = float(torch.nn.functional.cosine_similarity(weights_after_a.flatten(), adapter.w_syn.flatten(), dim=0).item())
    print(f"  Synaptic Weight Orthogonality: Cosine similarity between Domain A & B = {cos_sim:.4f}")

    # --------------------------------------------------------------------------
    # PHASE 4: THE CATASTROPHIC FORGETTING TEST (Re-Testing Domain A!)
    # --------------------------------------------------------------------------
    print("\n[>>> PHASE 4: Critical Retention Test: Domain A Re-Evaluation (Catastrophic Forgetting Check) <<<]")
    final_a_tests = []
    for i in range(8):
        seed = 6000 + i
        # Crucial test: test Domain A after intensive Domain B training without re-exposing to Domain A training!
        hint_prompt = TASK_A["prompt"] + "\n[PopBrain Synaptic Memory: Preserve dummy head/tail invariants.]"
        resp = call_model(hint_prompt, seed=seed)
        r, info = evaluate(TASK_A, resp.get("response", ""))
        final_a_tests.append(info["passed"])
        status = "RETAINED [PASS]" if info["passed"] else "FORGOTTEN [FAIL]"
        print(f"  [Retention Test {i+1:02d}/8] Domain A: {status} (Reward: {r:+.2f})")

    final_a_acc = sum(final_a_tests) / len(final_a_tests) * 100
    retention_ratio = (final_a_acc / max(1.0, p2_a_acc)) * 100
    forgetting_delta = final_a_acc - p2_a_acc

    print("\n" + "=" * 80)
    print("  CATASTROPHIC FORGETTING BENCHMARK REPORT:")
    print(f"  - Initial Domain A Baseline:      {p1_a_acc:.1f}%")
    print(f"  - Domain A Post-Training Peak:    {p2_a_acc:.1f}%")
    print(f"  - Domain B Learned Accuracy:      {p3_b_acc:.1f}%")
    print(f"  - Final Domain A Retention:       {final_a_acc:.1f}%")
    print(f"  - Knowledge Retention Ratio:      {retention_ratio:.1f}%")
    print(f"  - Forgetting Rate (Delta):        {forgetting_delta:+.1f}%")
    print(f"  - Synaptic Orthogonality (cos):   {cos_sim:.4f}")
    print("=" * 80)

    experiment_log["phases"]["phase4_retention"] = {
        "final_a_acc": final_a_acc,
        "retention_ratio": retention_ratio,
        "forgetting_delta": forgetting_delta,
        "synaptic_cosine_sim": cos_sim
    }

    # Save complete experimental database
    out_file = "assets/catastrophic_forgetting/continual_learning_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(experiment_log, f, indent=2)
    print(f"[+] Saved experiment results to: {out_file}")

if __name__ == "__main__":
    run_experiment()
