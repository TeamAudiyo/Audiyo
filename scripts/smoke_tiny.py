from __future__ import annotations

import json
import os
import time

import numpy as np


def main():
    import soundfile as sf

    from audiyo.chunked import compare_full_vs_tiled
    from audiyo.datasets import build_dataset
    from audiyo.testkit import TinyPipeline, build_tiny_model, count_parameters
    from audiyo.training import run_lora_training
    import torch

    print("Params:")
    print(json.dumps(count_parameters(TinyPipeline()), indent=1))

    model = build_tiny_model()
    t0 = time.perf_counter()
    result = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=8)
    gen_s = time.perf_counter() - t0
    result.save("tiny_demo.wav")
    print("Generated tiny_demo.wav shape " + str(tuple(result.waveform.shape)) + " peak " + str(round(result.peak, 4)) + " in " + str(round(gen_s, 2)) + " s")

    pipe = model.pipeline
    pipe.vae.eval()
    latents = torch.randn(1, 16, 352)
    rep = compare_full_vs_tiled(pipe, latents)
    print("Tiled vs full max diff " + str(rep["max_abs_diff"]))

    d = "tiny_smoke_data"
    os.makedirs(d, exist_ok=True)
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(os.path.join(d, "a.wav"), np.stack([tone, tone], axis=1), sr)
    with open(os.path.join(d, "a.txt"), "w", encoding="utf-8") as f:
        f.write("a test tone")
    ds = build_dataset(d, duration_seconds=2.0)
    report = run_lora_training(
        TinyPipeline(),
        ds,
        output_dir="tiny_smoke_out",
        max_steps=40,
        rank=4,
        alpha=4,
        target_modules=["to_q", "to_k", "to_v"],
        learning_rate=5e-4,
        device="cpu",
        base_model_id="tiny",
        duration_seconds=2.0,
        progress=True,
    )
    print("First-5 loss " + str(round(float(np.mean(report.losses[:5])), 4)) + " last-5 " + str(round(float(np.mean(report.losses[-5:])), 4)))
    print("Checks " + str(report.checks))


if __name__ == "__main__":
    main()
