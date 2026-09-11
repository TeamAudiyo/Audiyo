from __future__ import annotations

import numpy as np
import torch

from ..errors import ValidationError


def ensure_stereo(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        wav = wav[None, :]
    if wav.shape[0] != 2 and wav.shape[-1] == 2:
        wav = wav.T
    if wav.shape[0] != 2:
        raise ValidationError("Expected stereo audio with shape (2, samples), got " + str(wav.shape) + ".")
    return wav.astype(np.float32)


def extract_first_waveform(audios) -> np.ndarray:
    if isinstance(audios, torch.Tensor):
        wav = audios[0].detach().cpu().float().numpy()
        return ensure_stereo(wav)
    first = np.asarray(audios[0], dtype=np.float32)
    return ensure_stereo(first)


def extract_waveform_at(audios, index: int) -> np.ndarray:
    if isinstance(audios, torch.Tensor):
        wav = audios[index].detach().cpu().float().numpy()
        return ensure_stereo(wav)
    item = np.asarray(audios[index], dtype=np.float32)
    return ensure_stereo(item)


def count_waveforms(audios) -> int:
    if isinstance(audios, torch.Tensor):
        return int(audios.shape[0])
    return len(audios)
