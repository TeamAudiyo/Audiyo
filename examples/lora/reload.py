from __future__ import annotations

import sys


def main():
    from audiyo import AudioModel

    adapter = sys.argv[1] if len(sys.argv) > 1 else "my_adapter/adapter"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Rain on a tin roof"
    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    meta = model.load_adapter(adapter)
    print("Loaded adapter for " + str(meta.get("base_model")))
    result = model.generate(prompt=prompt, duration_seconds=8, seed=7)
    result.save("adapter_demo.wav")
    print("Wrote adapter_demo.wav, peak " + str(round(result.peak, 4)))


if __name__ == "__main__":
    main()
