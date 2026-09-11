from __future__ import annotations

import os
import shutil

import numpy as np


def main():
    import soundfile as sf

    from audiyo.datasets import build_dataset
    from audiyo.testkit import build_tiny_model
    from audiyo.training import run_lora_training
    import torch

    outdir = "listen"
    os.makedirs(outdir, exist_ok=True)
    data = os.path.join(outdir, "tone_data")
    os.makedirs(data, exist_ok=True)
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(os.path.join(data, "a.wav"), np.stack([tone, tone], axis=1), sr)
    with open(os.path.join(data, "a.txt"), "w", encoding="utf-8") as f:
        f.write("a pure test tone")

    model = build_tiny_model()
    base = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=8)
    base.save(os.path.join(outdir, "baseline_seed42.wav"))
    again = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=8)
    det = float(np.max(np.abs(base.waveform - again.waveform)))
    print("determinism max diff: " + str(det))

    ds = build_dataset(data, duration_seconds=2.0)
    adapter_out = os.path.join(outdir, "adapter_run")
    shutil.rmtree(adapter_out, ignore_errors=True)
    report = run_lora_training(
        model.pipeline,
        ds,
        output_dir=adapter_out,
        max_steps=40,
        rank=4,
        alpha=4,
        target_modules=["to_q", "to_k", "to_v"],
        learning_rate=5e-4,
        device="cpu",
        base_model_id="audiyo-testkit-tiny",
        duration_seconds=2.0,
        progress=False,
    )
    print("first-5 " + str(round(float(np.mean(report.losses[:5])), 4)) + " last-5 " + str(round(float(np.mean(report.losses[-5:])), 4)))

    model.load_adapter(os.path.join(outdir, "adapter_run", "adapter"))
    adapted = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=8)
    adapted.save(os.path.join(outdir, "adapter_seed42.wav"))
    shift = float(np.max(np.abs(base.waveform - adapted.waveform)))
    print("adapter shift max diff: " + str(shift))
    print("base peak " + str(round(base.peak, 4)) + " adapted peak " + str(round(adapted.peak, 4)))


if __name__ == "__main__":
    main()
