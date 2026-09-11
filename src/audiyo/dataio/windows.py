from __future__ import annotations

import numpy as np


def samples_for_duration(duration_s: float, sr: int = 44100) -> int:
    return int(round(duration_s * sr))


def crop_or_pad(wav: np.ndarray, target_samples: int, random_crop: bool = False, rng=None):
    """Fit audio to the model window. Short clips are silence-padded and the true length travels separately as seconds_total."""
    n = wav.shape[1]
    if n == target_samples:
        return wav, target_samples
    if n > target_samples:
        if random_crop:
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, n - target_samples + 1))
        else:
            start = 0
        return wav[:, start:start + target_samples], target_samples
    pad = np.zeros((wav.shape[0], target_samples - n), dtype=np.float32)
    out = np.concatenate([wav, pad], axis=1)
    return out, n
