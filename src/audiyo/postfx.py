from __future__ import annotations

import numpy as np

from .errors import ValidationError


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def validate_postfx(
    fade_in_ms: float = 0.0,
    fade_out_ms: float = 0.0,
    normalize_peak: float | None = None,
    limiter: bool = False,
    trim_silence: bool = False,
    trim_db: float = -50.0,
    duration_seconds: float | None = None,
) -> dict:
    if fade_in_ms is None:
        fade_in_ms = 0.0
    if fade_out_ms is None:
        fade_out_ms = 0.0
    if not isinstance(fade_in_ms, (int, float)) or not isinstance(fade_out_ms, (int, float)):
        raise ValidationError("fade lengths must be numbers in milliseconds.")
    if float(fade_in_ms) < 0 or float(fade_out_ms) < 0:
        raise ValidationError("fade lengths must be zero or more.")
    if float(fade_in_ms) > 5000 or float(fade_out_ms) > 10000:
        raise ValidationError("fade lengths look too long, keep fades short.")
    if normalize_peak is not None:
        try:
            target = float(normalize_peak)
        except Exception:
            raise ValidationError("normalize_peak must be a number between 0.1 and 1.0 or empty.")
        if not 0.1 <= target <= 1.0:
            raise ValidationError("normalize_peak must be between 0.1 and 1.0.")
    if duration_seconds is not None:
        total_ms = float(duration_seconds) * 1000.0
        if float(fade_in_ms) + float(fade_out_ms) >= total_ms:
            raise ValidationError("fades are longer than the clip itself.")
    if not isinstance(trim_db, (int, float)) or not -80.0 <= float(trim_db) <= -20.0:
        raise ValidationError("trim_db must be between -80 and -20.")
    return {
        "fade_in_ms": float(fade_in_ms),
        "fade_out_ms": float(fade_out_ms),
        "normalize_peak": None if normalize_peak is None else float(normalize_peak),
        "limiter": bool(limiter),
        "trim_silence": bool(trim_silence),
        "trim_db": float(trim_db),
    }


def apply_fade(wav: np.ndarray, sample_rate: int, fade_in_ms: float = 0.0, fade_out_ms: float = 0.0) -> np.ndarray:
    out = wav.astype(np.float32, copy=True)
    total = int(out.shape[1])
    if total <= 0:
        return out
    fade_in_n = int(round(float(fade_in_ms) * float(sample_rate) / 1000.0))
    fade_out_n = int(round(float(fade_out_ms) * float(sample_rate) / 1000.0))
    fade_in_n = max(0, min(fade_in_n, total))
    fade_out_n = max(0, min(fade_out_n, total))
    if fade_in_n > 1:
        ramp = np.linspace(0.0, 1.0, fade_in_n, dtype=np.float32)
        out[:, :fade_in_n] = out[:, :fade_in_n] * ramp[None, :]
    if fade_out_n > 1:
        ramp = np.linspace(1.0, 0.0, fade_out_n, dtype=np.float32)
        out[:, total - fade_out_n :] = out[:, total - fade_out_n :] * ramp[None, :]
    return out.astype(np.float32)


def normalize_to_peak(wav: np.ndarray, target: float = 0.89) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 1e-9:
        return wav.astype(np.float32, copy=False)
    gain = float(target) / peak
    return (wav.astype(np.float32) * gain).astype(np.float32)


def soft_limiter(wav: np.ndarray, drive: float = 1.5) -> np.ndarray:
    x = wav.astype(np.float32)
    y = np.tanh(x * float(drive))
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = (y / peak).astype(np.float32)
    return y.astype(np.float32)


def trim_edge_silence(wav: np.ndarray, sample_rate: int, threshold_db: float = -50.0) -> tuple[np.ndarray, dict]:
    info = {"trimmed_start": 0, "trimmed_end": 0, "threshold_db": float(threshold_db)}
    if wav.size == 0:
        return wav.astype(np.float32, copy=False), info
    thresh = _db_to_amp(float(threshold_db))
    loud = np.max(np.abs(wav), axis=0) > thresh
    if not np.any(loud):
        return wav.astype(np.float32, copy=False), info
    idx = np.where(loud)[0]
    start = int(idx[0])
    end = int(idx[-1]) + 1
    info["trimmed_start"] = start
    info["trimmed_end"] = int(wav.shape[1] - end)
    return wav[:, start:end].astype(np.float32), info


def apply_postfx(
    wav: np.ndarray,
    sample_rate: int,
    fade_in_ms: float = 0.0,
    fade_out_ms: float = 0.0,
    normalize_peak: float | None = None,
    limiter: bool = False,
    trim_silence: bool = False,
    trim_db: float = -50.0,
) -> tuple[np.ndarray, dict]:
    opts = validate_postfx(fade_in_ms, fade_out_ms, normalize_peak, limiter, trim_silence, trim_db)
    out = wav.astype(np.float32, copy=True)
    trim_info = {"trimmed_start": 0, "trimmed_end": 0, "threshold_db": float(trim_db)}
    if opts["trim_silence"]:
        out, trim_info = trim_edge_silence(out, sample_rate, opts["trim_db"])
    if opts["fade_in_ms"] or opts["fade_out_ms"]:
        out = apply_fade(out, sample_rate, opts["fade_in_ms"], opts["fade_out_ms"])
    if opts["limiter"]:
        out = soft_limiter(out)
    if opts["normalize_peak"] is not None:
        out = normalize_to_peak(out, opts["normalize_peak"])
    applied = {
        "fade_in_ms": opts["fade_in_ms"],
        "fade_out_ms": opts["fade_out_ms"],
        "normalize_peak": opts["normalize_peak"],
        "limiter": opts["limiter"],
        "trim_silence": opts["trim_silence"],
        "trim_db": opts["trim_db"],
        "trimmed_start": int(trim_info["trimmed_start"]),
        "trimmed_end": int(trim_info["trimmed_end"]),
        "peak_after": float(np.max(np.abs(out))) if out.size else 0.0,
    }
    return out.astype(np.float32), applied
