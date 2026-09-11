from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import MEMORY_MODES
from .errors import ValidationError
from .memopt import PRESET_DOCS, describe_presets
from .memopt.applier import apply_attention_slicing, apply_offload, apply_vae_flags


@dataclass
class AppliedMemoryConfig:
    mode: str
    dtype: str
    offload: str
    attention_slicing: bool
    vae_slicing: bool
    vae_tiling: bool
    device: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dtype": self.dtype,
            "offload": self.offload,
            "attention_slicing": self.attention_slicing,
            "vae_slicing": self.vae_slicing,
            "vae_tiling": self.vae_tiling,
            "device": self.device,
            "notes": self.notes,
        }


def _torch_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValidationError(f"Unknown dtype {name!r}. Use float32, float16, or bfloat16.")
    return mapping[name]


def resolve_dtype(requested: str | None, device: str, bf16_supported: bool) -> str:
    if requested is not None:
        if requested == "float16" and device == "cpu":
            raise ValidationError(
                "float16 on CPU is not supported by this pipeline and usually runs "
                "slower, not faster. Use float32 on CPU or move to CUDA."
            )
        return requested
    from .hardware import default_dtype

    return default_dtype(device, bf16_supported)


def apply_memory_preset(
    pipeline,
    mode: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    bf16_supported: bool = False,
    vae_tiling: bool | None = None,
    attention_slicing: bool | None = None,
) -> AppliedMemoryConfig:
    """Apply a preset to an already loaded pipeline. Returns what was done."""
    if mode not in MEMORY_MODES:
        raise ValidationError(
            f"Unknown memory_mode {mode!r}. Choose one of {list(MEMORY_MODES)}."
        )
    spec = PRESET_DOCS[mode]
    notes: list[str] = []

    use_attention_slicing = attention_slicing if attention_slicing is not None else spec["attention_slicing"]
    use_vae_tiling = vae_tiling if vae_tiling is not None else spec["vae_tiling"]
    use_vae_slicing = spec["vae_slicing"]

    offload = spec["offload"]
    apply_offload(pipeline, offload, device, notes)
    apply_attention_slicing(pipeline, bool(use_attention_slicing), notes)
    apply_vae_flags(pipeline, bool(use_vae_slicing), bool(use_vae_tiling), notes)

    resolved_dtype = resolve_dtype(dtype, device, bf16_supported)
    return AppliedMemoryConfig(
        mode=mode,
        dtype=resolved_dtype,
        offload=offload if device == "cuda" else "none",
        attention_slicing=bool(use_attention_slicing),
        vae_slicing=bool(use_vae_slicing),
        vae_tiling=bool(use_vae_tiling),
        device=device,
        notes=notes,
    )
