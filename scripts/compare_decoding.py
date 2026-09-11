from __future__ import annotations

import sys


def main():
    from audiyo import AudioModel
    from audiyo.chunked import compare_full_vs_tiled
    from audiyo.training import encode_audio_latents
    import torch

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    pipe = model.pipeline
    latents = torch.randn(1, 64, 128)
    report = compare_full_vs_tiled(pipe, latents)
    print("max abs diff: " + str(report["max_abs_diff"]))
    print("mean abs diff: " + str(report["mean_abs_diff"]))
    print("note: listen before keeping tiled output")


if __name__ == "__main__":
    main()
