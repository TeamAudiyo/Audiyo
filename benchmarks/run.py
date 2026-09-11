from __future__ import annotations

import json
import sys


def main():
    from audiyo import AudioModel
    from audiyo.benchmark import run_benchmark

    mode = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    prompts = ["Rain against a window with distant thunder"]
    model = AudioModel.from_pretrained(device="auto", memory_mode=mode)
    report = run_benchmark(model, prompts, duration_seconds=duration, num_inference_steps=20, repeat=2)
    with open("benchmark_" + mode + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print("Wrote benchmark_" + mode + ".json")


if __name__ == "__main__":
    main()
