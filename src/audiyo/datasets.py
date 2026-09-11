from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import load_audio_mono_stereo
from .dataio import crop_or_pad, samples_for_duration
from .config import SAMPLE_RATE
from .errors import ValidationError

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")


@dataclass
class ClipItem:
    audio_path: str | None
    waveform: np.ndarray | None
    caption: str
    seconds_total: float
    source: str = ""


@dataclass
class DatasetSummary:
    num_train: int
    num_val: int
    sample_rate: int
    window_samples: int
    deterministic: bool
    cached: bool
    cache_note: str = ""


def _hash_config(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def cache_key(checkpoint: str, preprocessing: dict[str, Any]) -> str:
    return _hash_config({"checkpoint": checkpoint, **preprocessing})


def discover_directory_items(root: str) -> list[tuple[str, str]]:
    if not os.path.isdir(root):
        raise ValidationError(f"Dataset directory not found: {root}")
    items: list[tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        base, ext = os.path.splitext(name)
        if ext.lower() not in AUDIO_EXTS:
            continue
        caption_path = os.path.join(root, base + ".txt")
        if not os.path.isfile(caption_path):
            raise ValidationError(
                f"Missing caption for {name}: expected {base}.txt next to the audio. "
                "Each audio file needs a same-basename text file with its caption."
            )
        with open(caption_path, encoding="utf-8") as f:
            caption = f.read().strip()
        if not caption:
            raise ValidationError(f"Caption file {caption_path} is empty.")
        items.append((os.path.join(root, name), caption))
    if not items:
        raise ValidationError(
            f"No audio files found in {root}. Expected .wav/.flac/.ogg/.mp3 with .txt captions."
        )
    return items


def load_csv_items(csv_path: str) -> list[tuple[str, str]]:
    if not os.path.isfile(csv_path):
        raise ValidationError(f"Dataset CSV not found: {csv_path}")
    base = os.path.dirname(os.path.abspath(csv_path))
    items: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "audio_path" not in reader.fieldnames or "caption" not in reader.fieldnames:
            raise ValidationError(
                f"CSV {csv_path} must have header columns audio_path,caption. "
                f"Found: {reader.fieldnames}."
            )
        for i, row in enumerate(reader):
            ap = row["audio_path"].strip()
            cap = row["caption"].strip()
            if not ap or not cap:
                raise ValidationError(f"Row {i} in {csv_path} has an empty path or caption.")
            if not os.path.isabs(ap):
                ap = os.path.join(base, ap)
            items.append((ap, cap))
    if not items:
        raise ValidationError(f"CSV {csv_path} contains no rows.")
    return items


def prepare_clip(
    audio_path: str,
    caption: str,
    *,
    duration_seconds: float,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> ClipItem:
    wav, _sr = load_audio_mono_stereo(audio_path)
    window = samples_for_duration(duration_seconds)
    true_samples = wav.shape[1]
    fitted, true_kept = crop_or_pad(wav, window, random_crop=random_crop, rng=rng)
    seconds_total = min(true_samples / SAMPLE_RATE, duration_seconds)
    _ = true_kept
    return ClipItem(audio_path=audio_path, waveform=fitted, caption=caption, seconds_total=float(seconds_total), source=audio_path)


@dataclass
class AudioCaptionDataset:
    items: list[ClipItem] = field(default_factory=list)
    window_samples: int = 0
    duration_seconds: float = 10.0
    deterministic: bool = True

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ClipItem:
        return self.items[idx]

    def split(self, validation_split: float, seed: int = 0) -> tuple["AudioCaptionDataset", "AudioCaptionDataset"]:
        if validation_split <= 0:
            empty = AudioCaptionDataset([], self.window_samples, self.duration_seconds, self.deterministic)
            return self, empty
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.items))
        n_val = max(1, int(round(len(self.items) * validation_split)))
        val_idx = set(order[:n_val].tolist())
        train = [c for i, c in enumerate(self.items) if i not in val_idx]
        val = [c for i, c in enumerate(self.items) if i in val_idx]
        kw = dict(window_samples=self.window_samples, duration_seconds=self.duration_seconds, deterministic=self.deterministic)
        return AudioCaptionDataset(train, **kw), AudioCaptionDataset(val, **kw)


def build_dataset(
    dataset: str,
    *,
    duration_seconds: float,
    deterministic: bool = True,
    seed: int = 0,
    text_column: str = "caption",
    audio_column: str = "audio",
    limit: int | None = None,
) -> AudioCaptionDataset:
    """Build an in-memory dataset. HF datasets load lazily per item at train time."""
    rng = np.random.default_rng(seed)
    window = samples_for_duration(duration_seconds)
    if os.path.isdir(dataset):
        pairs = discover_directory_items(dataset)
        if limit:
            pairs = pairs[:limit]
        clips = [
            prepare_clip(p, c, duration_seconds=duration_seconds, random_crop=not deterministic, rng=rng)
            for p, c in pairs
        ]
        return AudioCaptionDataset(clips, window, duration_seconds, deterministic)
    if os.path.isfile(dataset) and dataset.lower().endswith(".csv"):
        pairs = load_csv_items(dataset)
        if limit:
            pairs = pairs[:limit]
        clips = [
            prepare_clip(p, c, duration_seconds=duration_seconds, random_crop=not deterministic, rng=rng)
            for p, c in pairs
        ]
        return AudioCaptionDataset(clips, window, duration_seconds, deterministic)
    raise ValidationError(
        f"Could not interpret dataset {dataset!r}. Pass a directory of audio + .txt "
        "captions, a CSV with audio_path,caption columns, or use "
        "build_hf_dataset() for a Hugging Face dataset name."
    )


def summarize(train: AudioCaptionDataset, val: AudioCaptionDataset, *, cached: bool, checkpoint: str, preprocessing: dict) -> DatasetSummary:
    note = ""
    if cached:
        note = (
            f"Latent/embedding cache on (key {cache_key(checkpoint, preprocessing)}). "
            "Cached crops are fixed and latents skip posterior resampling: faster "
            "epochs, slightly less stochasticity."
        )
    return DatasetSummary(
        num_train=len(train),
        num_val=len(val),
        sample_rate=SAMPLE_RATE,
        window_samples=train.window_samples,
        deterministic=train.deterministic,
        cached=cached,
        cache_note=note,
    )
