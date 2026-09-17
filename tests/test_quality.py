from __future__ import annotations

import numpy as np


def _stereo_tone(sr=44100, seconds=1.0, freq=440.0):
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / float(sr)
    mono = np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)
    return np.stack([mono, mono]), sr


def test_silence_metrics():
    from audiyo.quality import analyze_waveform

    wav = np.zeros((2, 44100), dtype=np.float32)
    m = analyze_waveform(wav, 44100)
    assert m["peak"] == 0.0
    assert m["silence_fraction"] == 1.0
    assert m["duration_s"] == 1.0


def test_tone_has_centroid_near_freq():
    from audiyo.quality import analyze_waveform

    wav, sr = _stereo_tone(freq=440.0)
    m = analyze_waveform(wav, sr)
    assert 300.0 < m["spectral_centroid_hz"] < 900.0
    assert m["stereo_correlation"] > 0.99
    assert m["peak"] > 0.9


def test_clipping_flagged():
    from audiyo.quality import analyze_waveform

    wav = np.ones((2, 1000), dtype=np.float32) * 1.2
    m = analyze_waveform(wav, 44100)
    assert m["clipped_fraction"] == 1.0


def test_compare_reports_deltas():
    from audiyo.quality import analyze_waveform, compare_metrics

    quiet = np.zeros((2, 8000), dtype=np.float32)
    loud_tone, _ = _stereo_tone(seconds=8000 / 44100)
    a = analyze_waveform(quiet, 44100)
    b = analyze_waveform(loud_tone * 0.5, 44100)
    d = compare_metrics(a, b)
    assert d["delta_peak"] > 0.3


def test_sheet_writes_csv_and_md(tmp_path):
    from audiyo.quality import write_listening_sheet

    entries = [{"file": "a.wav", "prompt": "rain", "seed": 1, "rating": 4, "notes": "good"}]
    csv_path = str(tmp_path / "sheet.csv")
    md_path = str(tmp_path / "sheet.md")
    assert write_listening_sheet(entries, csv_path) == csv_path
    assert write_listening_sheet(entries, md_path) == md_path
