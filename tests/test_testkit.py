from __future__ import annotations

import numpy as np
import torch


def test_param_budget():
    from audiyo.testkit import TinyPipeline, count_parameters

    parts = count_parameters(TinyPipeline())
    assert 2000000 < parts["total"] < 8000000
    assert parts["transformer"] > parts["vae"]


def test_default_targets_hit_tiny_dit():
    from audiyo.lora import find_lora_targets
    from audiyo.testkit import TinyDiT

    found = find_lora_targets(TinyDiT())
    assert found == ["to_q", "to_k", "to_v", "to_out.0"]


def test_tiny_generate():
    from audiyo.testkit import build_tiny_model

    model = build_tiny_model()
    result = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=4)
    assert result.waveform.shape == (2, 88200)
    assert result.sample_rate == 44100
    assert bool(np.isfinite(result.waveform).all())
    assert result.peak > 0
    assert result.performance["latency_s"] >= 0


def test_tiny_save_roundtrip(tmp_path):
    import soundfile as sf

    from audiyo.audio import load_audio_mono_stereo
    from audiyo.testkit import build_tiny_model

    model = build_tiny_model()
    result = model.generate(prompt="cafe hum", duration_seconds=2, seed=7, num_inference_steps=2)
    path = str(tmp_path / "tiny.wav")
    result.save(path)
    wav, sr = load_audio_mono_stereo(path)
    assert sr == 44100
    assert wav.shape == result.waveform.shape
    info = sf.info(path)
    assert info.samplerate == 44100
    assert info.channels == 2


def test_tiny_deterministic():
    from audiyo.testkit import build_tiny_model

    model = build_tiny_model()
    first = model.generate(prompt="rain", duration_seconds=2, seed=42, num_inference_steps=2)
    second = model.generate(prompt="rain", duration_seconds=2, seed=42, num_inference_steps=2)
    assert float(np.max(np.abs(first.waveform - second.waveform))) == 0.0


def test_tiny_reload_after_train():
    import warnings

    import soundfile as sf

    from audiyo.datasets import build_dataset
    from audiyo.testkit import build_tiny_model
    from audiyo.training import run_lora_training

    import tempfile, os
    d = tempfile.mkdtemp()
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(os.path.join(d, "a.wav"), np.stack([tone, tone], axis=1), sr)
    with open(os.path.join(d, "a.txt"), "w", encoding="utf-8") as f:
        f.write("a test tone")
    ds = build_dataset(d, duration_seconds=2.0)
    model = build_tiny_model()
    out = os.path.join(d, "out")
    run_lora_training(
        model.pipeline,
        ds,
        output_dir=out,
        max_steps=2,
        rank=2,
        alpha=2,
        target_modules=["to_q", "to_k", "to_v"],
        learning_rate=5e-4,
        device="cpu",
        base_model_id="audiyo-testkit-tiny",
        duration_seconds=2.0,
        progress=False,
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model.load_adapter(os.path.join(out, "adapter"))
    nesting = [x for x in rec if "peft_config" in str(x.message)]
    assert nesting == []
    keys = list(model.pipeline.transformer.state_dict().keys())
    assert not any("base_model.model.base_model" in k for k in keys)
    result = model.generate(prompt="rain", duration_seconds=2, seed=1, num_inference_steps=2)
    assert bool(np.isfinite(result.waveform).all())


def test_tiny_tiling_matches():
    from audiyo.chunked import compare_full_vs_tiled
    from audiyo.testkit import TinyPipeline

    pipe = TinyPipeline()
    pipe.vae.eval()
    latents = torch.randn(1, 16, 352)
    report = compare_full_vs_tiled(pipe, latents)
    assert report["full_shape"] == report["tiled_shape"] == [1, 2, 90112]
    assert report["max_abs_diff"] < 1e-4


def test_tiny_overfit():
    import soundfile as sf

    from audiyo.datasets import build_dataset
    from audiyo.lora import load_adapter_into, read_adapter_meta
    from audiyo.testkit import TinyDiT, TinyPipeline
    from audiyo.training import run_lora_training

    import tempfile, os
    d = tempfile.mkdtemp()
    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr)).astype(np.float32)
    sf.write(os.path.join(d, "a.wav"), np.stack([tone, tone], axis=1), sr)
    with open(os.path.join(d, "a.txt"), "w", encoding="utf-8") as f:
        f.write("a test tone")
    ds = build_dataset(d, duration_seconds=2.0)
    pipe = TinyPipeline()
    out = os.path.join(d, "out")
    report = run_lora_training(
        pipe,
        ds,
        output_dir=out,
        max_steps=40,
        rank=4,
        alpha=4,
        target_modules=["to_q", "to_k", "to_v"],
        learning_rate=5e-4,
        device="cpu",
        base_model_id="tiny",
        duration_seconds=2.0,
        progress=False,
    )
    first = float(np.mean(report.losses[:5]))
    last = float(np.mean(report.losses[-5:]))
    assert last < first
    assert report.checks["finite_loss"] is True
    assert report.checks["nonzero_adapter_grads"] is True
    assert report.checks["adapter_params_updated"] is True
    assert report.checks["adapter_saved"] is True
    assert report.checks["resumable_checkpoint"] is True
    meta = read_adapter_meta(os.path.join(out, "adapter"))
    assert meta["base_model"] == "tiny"
    loaded = load_adapter_into(TinyDiT(), os.path.join(out, "adapter"))
    assert loaded is not None
