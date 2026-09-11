from __future__ import annotations

import json
import sys


def main():
    from audiyo import AudioModel
    from audiyo.benchmark import run_benchmark

    prompts = ["Rain against a window with distant thunder", "A busy cafe with cups clinking"]
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    report = run_benchmark(model, prompts, duration_seconds=duration, num_inference_steps=20, repeat=2)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
