"""
PopBrain Multi-Model Large LLM Benchmark & Comparative Research
================================================================
Evaluates 70B, 32B, 14B, and 3B models across identical standardized prompts:
1. Advanced Python Concurrency & Architecture (PriorityWorkerPool with backoff)
2. Closed SDK RAG Grounding & Zero-Shot API Extraction (BioReservoirEngine)
3. Systems & CUDA C++ Optimization (Warp Shuffle Reduction)

Models Evaluated:
- DeepSeek-R1-Distill-Llama-70B (IQ2_M, Hybrid 32 GPU / 48 CPU)
- Qwen-2.5-32B (Q2_K, Hybrid 51 GPU / 13 CPU)
- Qwen-2.5-Coder-14B-Q8 (Q8_0, Hybrid 28 GPU / 20 CPU)
- Qwen-2.5-Coder-14B-Q4 (Q4_K_M, 100% VRAM)
- Qwen-2.5-Coder-3B-Q4 (Q4_K_M, 100% VRAM)

Methodology:
- Deterministic seed = 42 for all models.
- Token budget: 1280 for DeepSeek-R1 (allows full <think> trace + complete code), 768 for dense coder models.
- Automatic result caching to allow incremental execution and instant resume.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
from typing import Dict, Any, List

BENCHMARK_PROMPTS = [
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

MODELS_CONFIG = [
    {
        "name": "deepseek-r1-70b-local",
        "display_name": "DeepSeek-R1-Distill-70B (IQ2_M)",
        "params": "70.6B",
        "format": "IQ2_M (2.73 BPW)",
        "execution": "Hybrid (32 GPU / 48 CPU)",
        "vram_gb": 10.1,
        "ram_gb": 19.8,
        "temperature": 0.6,
        "top_p": 0.95,
        "num_predict": 1280
    },
    {
        "name": "qwen2.5-32b-local",
        "display_name": "Qwen-2.5-32B (Q2_K)",
        "params": "32.5B",
        "format": "Q2_K (2.65 BPW)",
        "execution": "Hybrid (51 GPU / 13 CPU)",
        "vram_gb": 11.9,
        "ram_gb": 8.5,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 768
    },
    {
        "name": "qwen2.5-coder-14b-q8",
        "display_name": "Qwen2.5-Coder-14B (Q8_0)",
        "params": "14.7B",
        "format": "Q8_0 (8.50 BPW)",
        "execution": "Hybrid (28 GPU / 20 CPU)",
        "vram_gb": 8.9,
        "ram_gb": 7.8,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 768
    },
    {
        "name": "qwen2.5-coder-14b-q4",
        "display_name": "Qwen2.5-Coder-14B (Q4_K_M)",
        "params": "14.7B",
        "format": "Q4_K_M (4.50 BPW)",
        "execution": "100% VRAM (48 GPU / 0 CPU)",
        "vram_gb": 9.6,
        "ram_gb": 0.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 768
    },
    {
        "name": "qwen2.5-coder-3b-q4",
        "display_name": "Qwen2.5-Coder-3B (Q4_K_M)",
        "params": "3.0B",
        "format": "Q4_K_M (4.50 BPW)",
        "execution": "100% VRAM (36 GPU / 0 CPU)",
        "vram_gb": 2.7,
        "ram_gb": 0.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 768
    },
    {
        "name": "qwen2.5-coder-3b-f16",
        "display_name": "Qwen2.5-Coder-3B (FP16 Unquantized)",
        "params": "3.0B",
        "format": "FP16 (16.0 BPW)",
        "execution": "100% VRAM (36 GPU / 0 CPU)",
        "vram_gb": 6.8,
        "ram_gb": 0.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "num_predict": 768
    }
]


def query_ollama(
    model: str,
    prompt: str,
    api_url: str = "http://localhost:11434/api/generate",
    seed: int = 42,
    num_predict: int = 768,
    temperature: float = 0.0,
    top_p: float = 1.0
) -> Dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "options": {
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict
        },
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0

    tokens = data.get("eval_count", 0)
    eval_duration_sec = data.get("eval_duration", 1) / 1e9
    speed = tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0

    response_text = data.get("response", "")
    
    # Parse think tag if present
    think_text = ""
    answer_text = response_text
    if "<think>" in response_text:
        parts = response_text.split("</think>")
        if len(parts) > 1:
            think_text = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].strip()
        else:
            think_text = response_text.replace("<think>", "").strip()
            answer_text = "[Thinking budget exhausted before </think>]"

    return {
        "text": response_text,
        "think_trace": think_text,
        "final_code": answer_text,
        "tokens": tokens,
        "speed": round(speed, 2),
        "duration": round(elapsed, 2)
    }


def run_benchmark_for_model(cfg: Dict[str, Any], api_url: str, cache_dir: str) -> List[Dict[str, Any]]:
    cache_file = os.path.join(cache_dir, f"{cfg['name'].replace(':', '_')}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) == len(BENCHMARK_PROMPTS):
                print(f"[+] Loaded existing cached results for {cfg['display_name']} from {cache_file}")
                return cached
        except Exception:
            pass

    print(f"\n[*] Evaluating model: {cfg['display_name']} (Seed: 42, Temp: {cfg['temperature']}, Tokens: {cfg['num_predict']})")
    results = []
    for idx, test in enumerate(BENCHMARK_PROMPTS):
        print(f"  [{idx+1}/{len(BENCHMARK_PROMPTS)}] Running: {test['category']} - '{test['title']}'...")
        res = query_ollama(
            model=cfg["name"],
            prompt=test["prompt"],
            api_url=api_url,
            seed=42,
            num_predict=cfg["num_predict"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"]
        )
        has_closed_think = "</think>" in res["text"] if "<think>" in res["text"] else True
        think_status = "Closed & Solved" if has_closed_think else "Still Thinking"
        print(f"     -> {res['tokens']} tokens in {res['duration']}s ({res['speed']} tok/s) | State: {think_status}")

        results.append({
            "test_id": test["id"],
            "category": test["category"],
            "title": test["title"],
            "prompt": test["prompt"],
            "model_name": cfg["name"],
            "display_name": cfg["display_name"],
            "speed": res["speed"],
            "duration": res["duration"],
            "tokens": res["tokens"],
            "has_closed_think": has_closed_think,
            "think_trace": res["think_trace"],
            "final_code": res["final_code"],
            "full_response": res["text"]
        })

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Model LLM Benchmark Suite (70B, 32B, 14B, 3B).")
    parser.add_argument("--api-url", type=str, default="http://localhost:11434/api/generate", help="API Endpoint")
    parser.add_argument("--output", type=str, default="assets/multi_model_large_benchmark.json", help="Summary JSON output")
    parser.add_argument("--only-model", type=str, default=None, help="Run a specific model name")
    args = parser.parse_args()

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "cache")

    print("=" * 80)
    print("  PopBrain Comprehensive LLM Evaluation Benchmark: 70B | 32B | 14B | 3B")
    print(f"  API Endpoint: {args.api_url}")
    print(f"  Target Output: {args.output}")
    print("=" * 80)

    all_results = {}
    for cfg in MODELS_CONFIG:
        if args.only_model and cfg["name"] != args.only_model:
            continue
        model_results = run_benchmark_for_model(cfg, args.api_url, cache_dir)
        all_results[cfg["name"]] = {
            "config": cfg,
            "tests": model_results
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"  ALL BENCHMARKS COMPLETED! Master report saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
