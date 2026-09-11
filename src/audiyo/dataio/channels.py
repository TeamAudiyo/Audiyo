from __future__ import annotations

import numpy as np

from ..errors import ValidationError


def convert_channels(wav: np.ndarray, target: int = 2) -> np.ndarray:
    ch = wav.shape[0]
    if ch == target:
        return wav
    if target == 2 and ch == 1:
        return np.repeat(wav, 2, axis=0)
    if target == 1 and ch == 2:
        return wav.mean(axis=0, keepdims=True)
    raise ValidationError("Cannot convert " + str(ch) + " channels to " + str(target) + ".")
