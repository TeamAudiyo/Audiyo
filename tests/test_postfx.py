from __future__ import annotations

import numpy as np


def _tone(sr=44100, seconds=1.0, freq=440.0):
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / float(sr)
    mono = np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)
    return np.stack([mono, mono])


def test_fade_ramps_edges():
    from audiyo.postfx import apply_fade

    wav = np.ones((2, 44100), dtype=np.float32)
    out = apply_fade(wav, 44100, fade_in_ms=10.0, fade_out_ms=10.0)
    assert out.shape == wav.shape
    assert float(out[0, 0]) < 0.05
    assert float(out[0, -1]) < 0.05
    assert float(out[0, 22050]) > 0.99


def test_normalize_hits_target():
    from audiyo.postfx import normalize_to_peak

    wav = _tone() * 0.2
    out = normalize_to_peak(wav, 0.89)
    assert abs(float(np.max(np.abs(out))) - 0.89) < 0.01


def test_limiter_caps_peak():
    from audiyo.postfx import soft_limiter

    wav = np.ones((2, 1000), dtype=np.float32) * 3.0
    out = soft_limiter(wav)
    assert float(np.max(np.abs(out))) <= 1.01


def test_trim_removes_edge_silence():
    from audiyo.postfx import trim_edge_silence

    sr = 44100
    tone = _tone(sr, 0.5)
    pad = np.zeros((2, 4410), dtype=np.float32)
    wav = np.concatenate([pad, tone, pad], axis=1)
    out, info = trim_edge_silence(wav, sr, threshold_db=-50.0)
    assert out.shape[1] < wav.shape[1]
    assert info["trimmed_start"] > 1000
    assert info["trimmed_end"] > 1000


def test_postfx_rejects_long_fade():
    from audiyo.errors import ValidationError
    from audiyo.postfx import validate_postfx

    try:
        validate_postfx(900.0, 900.0, None, False, False, -50.0, 1.0)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_generation_config_accepts_postfx():
    from audiyo.config import GenerationConfig

    cfg = GenerationConfig(prompt="rain", duration_seconds=5, fade_in_ms=20.0, fade_out_ms=200.0, normalize_peak=0.89)
    cfg.validate(max_duration=47.55)
    assert cfg.normalize_peak == 0.89
