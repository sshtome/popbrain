import json

with open("assets/multi_model_large_benchmark.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 110)
print(f"{'Model':<32} | {'Format':<14} | {'Execution / Offload':<26} | {'Tok/s (avg)':<12} | {'Closed Think'}")
print("=" * 110)

for model_key, model_data in data.items():
    cfg = model_data["config"]
    tests = model_data["tests"]
    avg_speed = sum(t["speed"] for t in tests) / len(tests)
    speeds_str = ", ".join(f"{t['speed']:.1f}" for t in tests)
    closed = [t.get("has_closed_think", True) for t in tests]
    print(f"{cfg['display_name']:<32} | {cfg['format']:<14} | {cfg['execution']:<26} | {avg_speed:<5.1f} ({speeds_str}) | {closed}")

print("=" * 110)
print("\n--- Detailed Test Code & Reasoning Breakdown ---")
for model_key, model_data in data.items():
    cfg = model_data["config"]
    print(f"\n>>> MODEL: {cfg['display_name']} ({cfg['format']})")
    for t in model_data["tests"]:
        code_len = len(t.get("final_code", ""))
        think_len = len(t.get("think_trace", ""))
        resp_len = len(t.get("full_response", ""))
        print(f"  - [{t['test_id']}] Tokens: {t['tokens']} | Duration: {t['duration']:.2f}s | Speed: {t['speed']:.2f} tok/s | Closed: {t['has_closed_think']}")
        print(f"    Think chars: {think_len} | Code chars: {code_len} | Total chars: {resp_len}")
