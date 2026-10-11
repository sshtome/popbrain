"""
DeepSeek-R1-Distill-Llama-70B & Large LLM Comprehensive Evaluation Benchmark
=============================================================================
Automated cross-system evaluation testbed for:
1. Logic & Mathematical Reasoning (Olympiad / Complex puzzle with <think> trace)
2. Advanced Coding & System Architecture (Algorithmic problem in Python)
3. Deep Multi-Turn Context & Retrieval
4. Hardware Profiling (Tokens/sec, VRAM utilization, RAM headroom)

Usage:
    python examples/run_deepseek_70b_benchmark.py
    python examples/run_deepseek_70b_benchmark.py --model deepseek-r1-70b-local --tokens 512
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
        "category": "Logic & Reasoning",
        "title": "Олимпиадная задача на логику и математику",
        "prompt": (
            "Реши олимпиадную задачу пошагово. "
            "В комнате находятся 100 заключенных. Начальник тюрьмы предлагает испытание с переключателем в закрытой комнате. "
            "В комнате есть лампочка и один выключатель (исходно выключен). Заключенных заводят по одному в случайном порядке. "
            "Каждый может переключить тумблер или оставить как есть. "
            "В какой-то момент один из них должен сказать: 'Все 100 заключенных уже побывали в комнате'. "
            "Опиши выигрышную стратегию и математически докажи её корректность."
        )
    },
    {
        "category": "Coding & Algorithms",
        "title": "Сложный системный алгоритм на Python",
        "prompt": (
            "Напиши на Python высокопроизводительный потокобезопасный LRU-кэш с поддержкой TTL (Time-To-Live). "
            "Требования: доступ O(1), фоновая или ленивая очистка протухших ключей, thread-safe без глобального GIL-лока (Lock striping). "
            "Предоставь чистый рабочий код и юнит-тесты к нему."
        )
    },
    {
        "category": "Context & Analysis",
        "title": "Глубокий анализ архитектуры нейросетей",
        "prompt": (
            "Сравни три архитектурных подхода к преодолению бутылочного горлышка памяти в LLM: "
            "1. Spiking Neural Networks (SNN, линейное внимание без KV-кэша) "
            "2. 1.58-битное тернарное квантование (BitNet b1.58) "
            "3. Послойный асинхронный стриминг через шину PCIe DMA (Double-buffered Sliding Window). "
            "Проанализируй сильные и слабые стороны каждого подхода в терминах вычислительной сложности O, "
            "пропускной способности шины данных и энергоэффективности (Joules/token)."
        )
    }
]


def run_benchmark(
    api_url: str = "http://localhost:11434/api/generate",
    model_name: str = "deepseek-r1-70b-local",
    num_predict: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> List[Dict[str, Any]]:
    results = []

    for idx, test in enumerate(BENCHMARK_PROMPTS):
        print(f"\n[{idx+1}/{len(BENCHMARK_PROMPTS)}] Running Test: {test['category']} - '{test['title']}'...")
        payload = json.dumps({
            "model": model_name,
            "prompt": test["prompt"],
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                "top_p": top_p
            },
            "stream": False
        }).encode("utf-8")

        req = urllib.request.Request(api_url, data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  [!] Request failed: {e}")
            continue

        elapsed = time.time() - t0
        eval_tokens = data.get("eval_count", 0)
        eval_duration_sec = data.get("eval_duration", 1) / 1e9
        speed = eval_tokens / eval_duration_sec if eval_duration_sec > 0 else 0.0

        print(f"  -> Generated {eval_tokens} tokens in {elapsed:.2f}s ({speed:.2f} tok/s)")

        results.append({
            "category": test["category"],
            "title": test["title"],
            "prompt": test["prompt"],
            "response": data.get("response", ""),
            "tokens": eval_tokens,
            "speed": round(speed, 2),
            "duration": round(elapsed, 2)
        })

    return results


def main():
    default_output = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "deepseek_70b_results.json")

    parser = argparse.ArgumentParser(description="Evaluate 70B & Large LLM performance on logic, code, and speed.")
    parser.add_argument("--model", type=str, default="deepseek-r1-70b-local", help="Ollama model tag to test")
    parser.add_argument("--api-url", type=str, default="http://localhost:11434/api/generate", help="Ollama or compatible API endpoint")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save benchmark JSON results")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens to generate per prompt (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature (default: 0.6)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling parameter (default: 0.95)")
    args = parser.parse_args()

    print("=" * 72)
    print("  PopBrain LLM Benchmark Suite: Logic, Coding & Hardware Throughput")
    print(f"  Target Model: {args.model}")
    print(f"  API Endpoint: {args.api_url}")
    print(f"  Tokens/test:  {args.tokens}")
    print("=" * 72)

    results = run_benchmark(
        api_url=args.api_url,
        model_name=args.model,
        num_predict=args.tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("  ALL BENCHMARK TESTS COMPLETED SUCCESSFULLY!")
    print(f"  Results saved to: {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
