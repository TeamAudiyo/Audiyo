from __future__ import annotations

from typing import Any

from .config import SUPPORTED_CHECKPOINTS, check_checkpoint
from .errors import AuthError, CheckpointError, DependencyError, DeviceError, ValidationError, auth_hint, scrub_text


def require_backend() -> None:
    try:
        import diffusers
        import transformers
    except ImportError as exc:
        raise DependencyError(
            "Audio generation needs torch, diffusers, transformers, safetensors and "
            "soundfile. Run pip install audiyo and retry."
        ) from exc


def pipeline_max_duration(pipeline) -> float:
    """Re-derive the duration limit from the loaded components."""
    try:
        sample_size = float(pipeline.transformer.config.sample_size)
        hop = float(pipeline.vae.config.hop_length) if hasattr(pipeline.vae.config, "hop_length") else 2048.0
        sr = float(pipeline.vae.config.sampling_rate) if hasattr(pipeline.vae.config, "sampling_rate") else 44100.0
        return sample_size * hop / sr
    except Exception:
        from .config import MAX_DURATION_SECONDS

        return MAX_DURATION_SECONDS


def load_pipeline(
    checkpoint: str,
    *,
    torch_dtype,
    device_map: Any | None = None,
    token: str | bool | None = None,
) -> Any:
    require_backend()
    check_checkpoint(checkpoint)
    from diffusers import StableAudioPipeline

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if token is not None:
        kwargs["token"] = token
    try:
        pipe = StableAudioPipeline.from_pretrained(checkpoint, **kwargs)
    except Exception as exc:
        text = scrub_text(str(exc))
        lowered = text.lower()
        if any(k in lowered for k in ("401", "403", "gated", "unauthorized", "license", "token", "login")):
            raise auth_hint(text) from exc
        if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
            from .errors import oom_hint

            raise oom_hint("while loading the checkpoint") from exc
        raise CheckpointError(
            f"Could not load checkpoint {checkpoint!r}: {text}\n"
            "Hint: check the id, your network connection, and available disk/RAM."
        ) from exc
    return pipe


def check_negative_prompt_supported(pipeline) -> bool:
    import inspect

    try:
        params = inspect.signature(pipeline.__call__).parameters
        return "negative_prompt" in params
    except Exception:
        return True
