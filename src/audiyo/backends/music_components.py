from __future__ import annotations

from typing import Any

FALLBACK_COMPONENTS = ("tokenizer", "language_model", "rvq_depth_decoder")


def missing_front_end(pipe: Any) -> list:
    missing = []
    for name in FALLBACK_COMPONENTS:
        try:
            value = getattr(pipe, name, None)
        except Exception:
            value = None
        if value is None:
            missing.append(name)
    return missing


def load_fallback_component(name: str, base_checkpoint: str, torch_dtype=None, token=None, device_map=None, quant_flags: dict | None = None):
    flags = dict(quant_flags or {})
    if name == "tokenizer":
        from transformers import AutoTokenizer

        kwargs: dict = {"subfolder": "tokenizer"}
        if token is not None:
            kwargs["token"] = token
        return AutoTokenizer.from_pretrained(base_checkpoint, **kwargs)
    if name == "language_model":
        from transformers import Qwen3ForCausalLM

        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if flags.get("load_in_8bit") or flags.get("load_in_4bit"):
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                from ..errors import DependencyError

                raise DependencyError("Quantized loads need a transformers version with BitsAndBytesConfig.") from exc
            if flags.get("load_in_8bit"):
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        if device_map is not None:
            kwargs["device_map"] = device_map
        if token is not None:
            kwargs["token"] = token
        return Qwen3ForCausalLM.from_pretrained(base_checkpoint, subfolder="language_model", **kwargs)
    if name == "rvq_depth_decoder":
        from diffusers import MiniMaxMusic3RVQDepthDecoder

        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        if token is not None:
            kwargs["token"] = token
        return MiniMaxMusic3RVQDepthDecoder.from_pretrained(base_checkpoint, subfolder="rvq_depth_decoder", **kwargs)
    from ..errors import ValidationError

    raise ValidationError("No fallback known for component " + repr(name) + ".")


def ensure_front_end(pipe: Any, base_checkpoint: str, torch_dtype=None, token=None, device_map=None, quant_flags: dict | None = None) -> list:
    from ..errors import CheckpointError, scrub_text

    fixed = []
    for name in missing_front_end(pipe):
        try:
            module = load_fallback_component(name, base_checkpoint, torch_dtype, token, device_map, quant_flags)
        except Exception as exc:
            raise CheckpointError(
                "Music pipeline is missing " + name + " and the fallback load from " + base_checkpoint + " failed: " + scrub_text(str(exc))[:300]
            ) from exc
        try:
            setattr(pipe, name, module)
        except Exception as exc:
            raise CheckpointError("Could not attach fallback " + name + ": " + scrub_text(str(exc))[:200]) from exc
        try:
            check = getattr(pipe, name, None)
        except Exception:
            check = None
        if check is None:
            raise CheckpointError("Fallback " + name + " did not stick on the pipeline.")
        fixed.append(name)
    return fixed
