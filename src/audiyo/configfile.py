from __future__ import annotations

import json
import os

from .errors import ValidationError

GENERATE_KEYS = (
    "prompt",
    "output",
    "duration",
    "seed",
    "steps",
    "guidance",
    "negative_prompt",
    "lyrics",
    "fade_in_ms",
    "fade_out_ms",
    "normalize_peak",
    "limiter",
    "trim_silence",
    "checkpoint",
    "device",
    "memory_mode",
    "dtype",
    "token",
)


def load_config_file(path: str) -> dict:
    if not path:
        raise ValidationError("config path must be a non-empty string.")
    if not os.path.isfile(path):
        raise ValidationError("config file not found: " + str(path))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ValidationError("could not read config file " + str(path) + ": " + str(exc)[:300]) from exc
    if not isinstance(data, dict):
        raise ValidationError("config file must hold a JSON object at the top level.")
    return data


def resolve_options(defaults: dict, cli: dict, config: dict) -> dict:
    merged = dict(defaults)
    for key, value in config.items():
        if key in defaults:
            merged[key] = value
    for key, value in cli.items():
        if key not in defaults:
            continue
        if value != defaults.get(key):
            merged[key] = value
    return merged


def filter_generate_config(data: dict) -> dict:
    return {k: v for k, v in data.items() if k in GENERATE_KEYS}


def write_example_config(path: str) -> str:
    example = {
        "prompt": "Rain against a window with distant thunder",
        "output": "output.wav",
        "duration": 10.0,
        "seed": 42,
        "steps": 100,
        "guidance": 7.0,
        "negative_prompt": None,
        "lyrics": None,
        "fade_in_ms": 20.0,
        "fade_out_ms": 200.0,
        "normalize_peak": None,
        "limiter": False,
        "trim_silence": False,
        "checkpoint": "stabilityai/stable-audio-open-1.0",
        "device": "auto",
        "memory_mode": "balanced",
        "dtype": None,
        "token": None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(example, handle, indent=2)
    return path
