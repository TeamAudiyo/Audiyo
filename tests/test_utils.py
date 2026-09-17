from __future__ import annotations

import numpy as np
import torch


def test_import_stays_light():
    import audiyo

    assert audiyo.__version__ == "0.3.2"
    model_cls = audiyo.AudioModel
    assert model_cls is not None


def test_inference_helpers():
    from audiyo.inference import ensure_stereo, make_cpu_generator

    gen = make_cpu_generator(42)
    assert gen is not None
    assert make_cpu_generator(None) is None
    wav = ensure_stereo(np.ones((2, 8), dtype=np.float32))
    assert wav.shape == (2, 8)


def test_benchutils_helpers():
    from audiyo.benchutils import mean_of, summarize_latencies

    assert mean_of([1.0, 3.0]) == 2.0
    summary = summarize_latencies([1.0, 1.0], 10.0)
    assert summary["audio_per_wall_s"] == 10.0


def test_v_loss_shapes():
    from audiyo.training import v_loss_terms

    x = torch.randn(1, 4, 8)
    e = torch.randn(1, 4, 8)
    t = torch.rand(1)
    noised, target = v_loss_terms(x, e, t)
    assert tuple(noised.shape) == (1, 4, 8)
    assert tuple(target.shape) == (1, 4, 8)
