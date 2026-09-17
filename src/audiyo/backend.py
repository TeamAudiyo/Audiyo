from __future__ import annotations

from typing import Any

from .backends import describe_backend, detect_model_type, load_pipeline
from .backends.base import require_backend
from .config import MAX_DURATION_SECONDS

__all__ = ["check_negative_prompt_supported", "describe_backend", "detect_model_type", "load_pipeline", "pipeline_max_duration", "require_backend"]


def pipeline_max_duration(pipeline) -> float:
    try:
        name = type(pipeline).__name__
        if "MiniMax" in name or hasattr(pipeline, "language_model"):
            return 360.0
        sample_size = float(pipeline.transformer.config.sample_size)
        hop = float(pipeline.vae.config.hop_length) if hasattr(pipeline.vae.config, "hop_length") else 2048.0
        sr = float(pipeline.vae.config.sampling_rate) if hasattr(pipeline.vae.config, "sampling_rate") else 44100.0
        return sample_size * hop / sr
    except Exception:
        return MAX_DURATION_SECONDS


def check_negative_prompt_supported(pipeline) -> bool:
    import inspect

    try:
        params = inspect.signature(pipeline.__call__).parameters
        return "negative_prompt" in params
    except Exception:
        return True
