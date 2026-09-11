from __future__ import annotations

import json
import sys


def main():
    from audiyo import AudioModel

    args = sys.argv[1:]
    prompt = args[0] if len(args) > 0 else "Rain against a window with distant thunder"
    out = args[1] if len(args) > 1 else "rain.wav"
    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    result = model.generate(prompt=prompt, duration_seconds=10, seed=42)
    result.save(out)
    print("Wrote " + out)
    print("Peak " + str(round(result.peak, 4)))
    print(json.dumps(result.performance, indent=2, default=str))


if __name__ == "__main__":
    main()
