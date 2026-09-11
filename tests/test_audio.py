from __future__ import annotations

import numpy as np


def test_audio_result_shape_guard():
    from audiyo.audio import AudioResult
    from audiyo.errors import ValidationError

    try:
        AudioResult(waveform=np.zeros((1, 100), dtype=np.float32), sample_rate=44100, prompt="x", duration_seconds=1.0)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_crop_pad_roundtrip():
    from audiyo.audio import crop_or_pad

    wav = np.ones((2, 100), dtype=np.float32)
    out, _ = crop_or_pad(wav, 200)
    assert out.shape == (2, 200)
    assert float(out[:, 100:].sum()) == 0.0
    out2, _ = crop_or_pad(np.ones((2, 300), dtype=np.float32), 200)
    assert out2.shape == (2, 200)


def test_convert_channels():
    from audiyo.audio import convert_channels

    mono = np.ones((1, 50), dtype=np.float32)
    stereo = convert_channels(mono, 2)
    assert stereo.shape == (2, 50)


def test_result_reports_clipping():
    from audiyo.audio import AudioResult

    wav = np.ones((2, 10), dtype=np.float32) * 1.5
    result = AudioResult(waveform=wav, sample_rate=44100, prompt="x", duration_seconds=1.0)
    assert result.peak == 1.5
    assert result.clipped_fraction == 1.0
