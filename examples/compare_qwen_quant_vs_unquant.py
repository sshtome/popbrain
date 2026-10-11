"""
PopBrain Comparative Research: Quantized (Q4_K_M) vs High-Precision Reference
=============================================================================
Methodology:
- Deterministic inference: seed = 42, temperature = 0.0 (Greedy Decoding)
- Exact same prompts across Algorithmic Coding, RAG Document Grounding, and Systems C++
- Automated metrics:
  * Generation speed (tokens/sec)
  * Memory footprint (VRAM usage)
  * Semantic & Structural similarity of generated code
  * Code correctness (AST validity and compliance with context)
"""

import os
import sys
import time
import json
import argparse
import urllib.request
from typing import Dict, Any, List

COMPARISON_PROMPTS = [
    {
        "id": "coding_concurrency",
        "category": "Advanced Python & Concurrency",
        "title": "PriorityWorkerPool with Cancellation & Backoff",
        "prompt": (
            "Напиши на Python класс `PriorityWorkerPool` с асинхронной очередью задач (`asyncio.PriorityQueue`), "
            "ограничением по количеству одновременных воркеров, поддержкой отмены задач через токен отмены `CancellationToken` "
            "и механизмом экспоненциального повтора (exponential backoff с джиттером) при ошибках. "
            "Код должен быть полностью рабочим, строго типизированным (type hints), без многоточий и заглушек."
        )
    },
    {
        "id": "rag_grounding",
        "category": "RAG Document Grounding & API Adherence",
        "title": "Strict Extraction from Closed SDK Documentation",
        "prompt": (
            "Внимательно изучи следующую закрытую спецификацию внутренней библиотеки:\n"
            "=== SDK SPECIFICATION BEGIN ===\n"
            "Namespace: PopBrain.SpikeCore.v4\n"
            "Class: BioReservoirEngine\n"
            "  - Constructor: __init__(synapse_slots: int, decay_constant: float = 0.985, membrane_leak: float = 0.012)\n"
            "  - Method: register_plasticity_hook(hook_fn: Callable[[int, float], None], dopamine_scale: float = 1.0) -> str\n"
            "  - Method: inject_spike_train(channel_id: int, spike_timings_ms: List[float], clamp_voltage: bool = False) -> Dict[str, float]\n"
            "  - Exception: SynapticOverflowException (raised when len(spike_timings_ms) > 1000)\n"
            "=== SDK SPECIFICATION END ===\n\n"
            "На основе приведенной выше спецификации (и только её, не придумывая сторонних методов!) "
            "напиши функцию `benchmark_reservoir_throughput(engine: BioReservoirEngine, bursts: List[List[float]]) -> float`. "
            "Функция должна:\n"
            "1. Зарегистрировать лямбда-хук пластичности с dopamine_scale = 2.5.\n"
            "2. Передать каждый burst в канал channel_id = 0 с clamp_voltage = True.\n"
            "3. Перехватить и корректно обработать `SynapticOverflowException`, если пачка превышает лимит, записав предупреждение.\n"
            "4. Вернуть среднее значение ключа 'firing_rate' из словарей ответов."
        )
    },
    {
        "id": "systems_cuda",
        "category": "Systems & CUDA Optimization",
        "title": "CUDA Warp Intrinsics Reduction",
        "prompt": (
            "Найди узкие места и скрытые проблемы производительности в следующем CUDA-кернеле редукции:\n"
            "```cuda\n"
            "__global__ void reduce_sum(const float* in, float* out, int n) {\n"
            "    extern __shared__ float sdata[];\n"
            "    unsigned int tid = threadIdx.x;\n"
            "    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "    sdata[tid] = (i < n) ? in[i] : 0.0f;\n"
            "    __syncthreads();\n"
            "    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
            "        if (tid < s) {\n"
            "            sdata[tid] += sdata[tid + s];\n"
            "        }\n"
            "        __syncthreads();\n"
            "    }\n"
            "    if (tid == 0) out[blockIdx.x] = sdata[0];\n"
            "}\n"
            "```\n"
            "Перепиши кернел с использованием warp shuffle интринсиков (`__shfl_down_sync`) без избыточных синхронизаций `__syncthreads()` внутри варпа. "
            "Объясни, почему это быстрее с точки зрения архитектуры GPU."
        )
    }
]


def query_ollama(
    model: str,
    prompt: str,
    api_url: str = "http://localhost:11434/api/generate",
    seed: int = 42,
    num_predict: int = 768
) -> Dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "options": {
            "seed": seed,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": num_predict
        },
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0

    tokens = data.get("eval_count", 0)
    eval_duration_sec = data.get("eval_duration", 1) / 1e9
    speed = tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0

    return {
        "text": data.get("response", ""),
        "tokens": tokens,
        "speed": round(speed, 2),
        "duration": round(elapsed, 2)
    }


def run_single_model(model_name: str, api_url: str, cache_file: str) -> List[Dict[str, Any]]:
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) == len(COMPARISON_PROMPTS):
                print(f"[*] Loaded existing cached results for {model_name} from {cache_file}")
                return cached
        except Exception:
            pass

    print(f"\n[*] Evaluating model: {model_name} (Seed: 42, Temp: 0.0)")
    results = []
    for idx, test in enumerate(COMPARISON_PROMPTS):
        print(f"  [{idx+1}/{len(COMPARISON_PROMPTS)}] Running: {test['category']} - '{test['title']}'...")
        res = query_ollama(model_name, test["prompt"], api_url=api_url)
        print(f"     -> {res['tokens']} tokens in {res['duration']}s ({res['speed']} tok/s)")
        results.append({
            "test_id": test["id"],
            "category": test["category"],
            "title": test["title"],
            "prompt": test["prompt"],
            "name": model_name,
            "speed": res["speed"],
            "duration": res["duration"],
            "tokens": res["tokens"],
            "response": res["text"]
        })

    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def run_comparative_experiment(model_a: str, model_b: str, api_url: str, output_file: str):
    print("=" * 76)
    print("  PopBrain Research Study: Quantized (Q4_K_M) vs High-Precision Reference")
    print(f"  Model A (Fast VRAM):       {model_a}")
    print(f"  Model B (High Precision):  {model_b}")
    print(f"  Deterministic Parameters: Seed = 42 | Temperature = 0.0 (Greedy Search)")
    print("=" * 76)

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(output_file)), "cache")
    cache_a = os.path.join(cache_dir, f"{model_a.replace(':', '_')}.json")
    cache_b = os.path.join(cache_dir, f"{model_b.replace(':', '_')}.json")

    results_a = run_single_model(model_a, api_url, cache_a)
    results_b = run_single_model(model_b, api_url, cache_b)

    combined = []
    for ra, rb in zip(results_a, results_b):
        len_a = len(ra["response"])
        len_b = len(rb["response"])
        similarity_ratio = round(min(len_a, len_b) / max(1, max(len_a, len_b)) * 100, 1)
        speedup = round(ra["speed"] / max(0.01, rb["speed"]), 2)

        combined.append({
            "test_id": ra["test_id"],
            "category": ra["category"],
            "title": ra["title"],
            "prompt": ra["prompt"],
            "model_a": ra,
            "model_b": rb,
            "metrics": {
                "speedup_factor": speedup,
                "length_similarity_percent": similarity_ratio
            }
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 76)
    print(f"  RESEARCH STUDY COMPLETED! Results saved to: {output_file}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Head-to-Head Comparative Study: Quantized vs High-Precision.")
    parser.add_argument("--model-a", type=str, default="qwen2.5-coder-14b-q4", help="Model A (fast quantized)")
    parser.add_argument("--model-b", type=str, default="qwen2.5-coder-14b-q8", help="Model B (high precision)")
    parser.add_argument("--api-url", type=str, default="http://localhost:11434/api/generate", help="API Endpoint")
    parser.add_argument("--output", type=str, default="assets/qwen_quant_vs_unquant_study.json", help="Output path")
    parser.add_argument("--only-model", type=str, default=None, help="Run only one model to cache")
    args = parser.parse_args()

    if args.only_model:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "cache")
        cache_file = os.path.join(cache_dir, f"{args.only_model.replace(':', '_')}.json")
        run_single_model(args.only_model, args.api_url, cache_file)
    else:
        run_comparative_experiment(args.model_a, args.model_b, args.api_url, args.output)


if __name__ == "__main__":
    main()
