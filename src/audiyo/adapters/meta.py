from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class AdapterMeta:
    base_model: str
    rank: int
    alpha: int
    target_modules: list
    audiyo_version: str = ""


def meta_path(adapter_dir: str) -> str:
    return os.path.join(adapter_dir, "audiyo_adapter.json")


def write_meta(adapter_dir: str, meta: AdapterMeta) -> str:
    from .._version import __version__

    os.makedirs(adapter_dir, exist_ok=True)
    meta.audiyo_version = meta.audiyo_version or __version__
    payload = {
        "base_model": meta.base_model,
        "rank": meta.rank,
        "alpha": meta.alpha,
        "target_modules": list(meta.target_modules),
        "audiyo_version": meta.audiyo_version,
        "license_note": "Adapter derived from stabilityai/stable-audio-open-1.0. Redistribution follows the Stability AI Community License.",
    }
    with open(meta_path(adapter_dir), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return adapter_dir


def read_meta(adapter_dir: str) -> dict:
    from ..errors import ValidationError

    path = meta_path(adapter_dir)
    if not os.path.isfile(path):
        raise ValidationError("No audiyo_adapter.json in " + str(adapter_dir) + ". Not an Audiyo adapter?")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
