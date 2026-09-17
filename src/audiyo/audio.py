from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import NUM_CHANNELS, SAMPLE_RATE
from .dataio import SUPPORTED_EXTENSIONS, convert_channels, crop_or_pad, samples_for_duration
from .errors import ValidationError


@dataclass
class AudioResult:
    waveform: np.ndarray
    sample_rate: int
    prompt: str
    duration_seconds: float
    seed: int | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.waveform.ndim != 2 or self.waveform.shape[0] != NUM_CHANNELS:
            raise ValidationError(
                f"waveform must have shape (2, samples), got {self.waveform.shape}."
            )

    @property
    def num_samples(self) -> int:
        return int(self.waveform.shape[1])

    @property
    def peak(self) -> float:
        if self.waveform.size == 0:
            return 0.0
        return float(np.max(np.abs(self.waveform)))

    @property
    def clipped_fraction(self) -> float:
        if self.waveform.size == 0:
            return 0.0
        over = np.sum(np.abs(self.waveform) >= 1.0)
        return float(over) / float(self.waveform.size)

    def save(self, path: str, normalize: bool = False) -> str:
        try:
            import soundfile as sf
        except ImportError as exc:
            from .errors import DependencyError

            raise DependencyError(
                "Saving audio needs the soundfile package. Install audiyo with "
                "pip install audiyo, which includes it."
            ) from exc
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".wav", ".flac", ".ogg", ".opus"):
            raise ValidationError(
                "unsupported output extension " + repr(ext) + ". Use wav, flac, ogg, or opus."
            )
        data = self.waveform
        if normalize:
            peak = self.peak
            if peak > 0:
                data = (data / max(peak, 1e-8) * 0.99).astype(np.float32)
        else:
            data = data.astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        sf.write(path, data.T, self.sample_rate)
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": int(self.waveform.shape[0]),
            "num_samples": self.num_samples,
            "duration_seconds": self.duration_seconds,
            "prompt": self.prompt,
            "seed": self.seed,
            "peak": self.peak,
            "clipped_fraction": self.clipped_fraction,
            "settings": self.settings,
            "performance": self.performance,
        }


def load_audio_mono_stereo(
    path: str, target_sr: int = SAMPLE_RATE, target_channels: int = NUM_CHANNELS
) -> tuple[np.ndarray, int]:
    """Load a file to (channels, samples) float32. Errors name the file."""
    if not os.path.isfile(path):
        raise ValidationError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported audio extension {ext!r} for {path}. "
            f"Supported: {list(SUPPORTED_EXTENSIONS)}."
        )
    try:
        import soundfile as sf
    except ImportError as exc:
        from .errors import DependencyError

        raise DependencyError("Loading audio needs the 'soundfile' package.") from exc
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
    except Exception as exc:
        raise ValidationError(f"Could not decode audio file {path}: {exc}") from exc
    if data.shape[0] == 0 or np.isnan(data).any() or np.isinf(data).any():
        raise ValidationError(f"Audio file {path} is empty or contains NaN/inf samples.")
    wav = data.T
    if sr != target_sr:
        wav = resample_audio(wav, sr, target_sr)
        sr = target_sr
    wav = convert_channels(wav, target_channels)
    return wav.astype(np.float32), sr


def resample_audio(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav
    try:
        import librosa
    except ImportError:
        raise ValidationError(
            "Resampling needs librosa (pip install audiyo[train]). "
            f"Cannot convert {src_sr} Hz to {dst_sr} Hz without it."
        )
    out = np.stack([librosa.resample(ch, orig_sr=src_sr, target_sr=dst_sr) for ch in wav])
    return out.astype(np.float32)
