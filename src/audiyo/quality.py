from __future__ import annotations

import csv
import math
import os

import numpy as np

from .errors import ValidationError


def _mono(wav: np.ndarray) -> np.ndarray:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 1:
        return x
    return x.mean(axis=0).astype(np.float32)


def analyze_waveform(wav: np.ndarray, sample_rate: int) -> dict:
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[0] not in (1, 2):
        raise ValidationError("waveform must have shape (channels, samples) with 1 or 2 channels.")
    if int(sample_rate) <= 0:
        raise ValidationError("sample_rate must be positive.")
    n = int(x.shape[1])
    duration = float(n) / float(sample_rate) if n else 0.0
    peak = float(np.max(np.abs(x))) if n else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
    if rms > 1e-12 and peak > 0:
        crest_db = float(20.0 * math.log10(peak / rms))
    else:
        crest_db = 0.0
    dc = float(np.mean(x.astype(np.float64))) if n else 0.0
    over = int(np.sum(np.abs(x) >= 1.0)) if n else 0
    clipped = float(over) / float(x.size) if x.size else 0.0
    thresh = 10.0 ** (-50.0 / 20.0)
    if n:
        quiet = np.max(np.abs(x), axis=0) < thresh
        silence = float(np.sum(quiet)) / float(n)
    else:
        silence = 1.0
    mono = _mono(x)
    if n > 1 and np.sum(np.abs(mono)) > 1e-12:
        mag = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
        total = float(np.sum(mag))
        if total > 1e-12:
            centroid = float(np.sum(freqs * mag) / total)
            spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / total))
            with np.errstate(divide="ignore"):
                logmag = np.log(mag + 1e-12)
            flatness = float(np.exp(np.mean(logmag)) / (np.mean(mag) + 1e-12))
        else:
            centroid = 0.0
            spread = 0.0
            flatness = 0.0
        signs = np.sign(mono)
        crossings = np.sum(np.abs(np.diff(signs)) > 0)
        zcr = float(crossings) / float(max(n - 1, 1))
    else:
        centroid = 0.0
        spread = 0.0
        flatness = 0.0
        zcr = 0.0
    if x.shape[0] == 2 and n > 1:
        left = x[0].astype(np.float64)
        right = x[1].astype(np.float64)
        left = left - left.mean()
        right = right - right.mean()
        denom = float(np.sqrt(np.sum(left ** 2) * np.sum(right ** 2)))
        if denom > 1e-12:
            corr = float(np.sum(left * right) / denom)
        else:
            corr = 1.0
    else:
        corr = 1.0
    return {
        "sample_rate": int(sample_rate),
        "channels": int(x.shape[0]),
        "num_samples": n,
        "duration_s": round(duration, 4),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "crest_db": round(crest_db, 3),
        "dc_offset": round(dc, 6),
        "clipped_fraction": round(clipped, 6),
        "silence_fraction": round(silence, 6),
        "spectral_centroid_hz": round(centroid, 2),
        "spectral_bandwidth_hz": round(spread, 2),
        "spectral_flatness": round(flatness, 6),
        "zero_crossing_rate": round(zcr, 6),
        "stereo_correlation": round(max(-1.0, min(1.0, corr)), 4),
    }


def analyze_file(path: str) -> dict:
    from .audio import load_audio_mono_stereo

    wav, sr = load_audio_mono_stereo(path)
    metrics = analyze_waveform(wav, sr)
    metrics["file"] = os.path.abspath(path)
    return metrics


def compare_metrics(before: dict, after: dict) -> dict:
    out = {}
    for key in ("peak", "rms", "crest_db", "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_flatness", "zero_crossing_rate", "stereo_correlation", "silence_fraction", "clipped_fraction"):
        if key in before and key in after:
            try:
                out["delta_" + key] = round(float(after[key]) - float(before[key]), 6)
            except Exception:
                continue
    return out


def compare_files(path_a: str, path_b: str) -> dict:
    a = analyze_file(path_a)
    b = analyze_file(path_b)
    deltas = compare_metrics(a, b)
    return {"file_a": a, "file_b": b, "deltas": deltas}


def write_listening_sheet(entries: list, path: str) -> str:
    if not isinstance(entries, list) or not entries:
        raise ValidationError("entries must be a non-empty list of dicts.")
    fields = ["file", "prompt", "seed", "rating", "notes"]
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("file"):
            raise ValidationError("each entry needs at least a file value.")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    lower = path.lower()
    if lower.endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for entry in entries:
                writer.writerow({k: entry.get(k, "") for k in fields})
        return path
    if lower.endswith(".md"):
        lines = []
        lines.append("Listening notes")
        lines.append("")
        lines.append("file | prompt | seed | rating | notes")
        lines.append("--- | --- | --- | --- | ---")
        for entry in entries:
            cells = [str(entry.get(k, "")).replace("|", "/").replace("\n", " ") for k in fields]
            lines.append(" | ".join(cells))
        lines.append("")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path
    raise ValidationError("sheet path must end with .csv or .md.")
