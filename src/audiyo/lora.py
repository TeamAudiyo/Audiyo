from __future__ import annotations

import os

import torch.nn as nn

from .adapters import AdapterMeta, DEFAULT_TARGET_MODULES, find_lora_targets, list_linear_modules, verify_targets_or_raise
from .adapters.meta import read_meta as read_adapter_meta, write_meta
from .errors import DependencyError, ValidationError


def require_peft() -> None:
    try:
        import peft
    except ImportError as exc:
        raise DependencyError(
            "LoRA needs the 'peft' and 'accelerate' packages. "
            "Run pip install audiyo[lora] and retry."
        ) from exc


def attach_lora(
    transformer: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 16,
    target_modules: tuple[str, ...] | list[str] | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    require_peft()
    from peft import LoraConfig, get_peft_model

    wanted = tuple(target_modules) if target_modules else DEFAULT_TARGET_MODULES
    matched = verify_targets_or_raise(transformer, wanted)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(matched),
        bias="none",
        task_type=None,
    )
    model = get_peft_model(transformer, config)
    return model


def trainable_summary(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def assert_only_adapter_trainable(pipeline_or_transformer: nn.Module) -> tuple[int, int]:
    trainable_names = [n for n, p in pipeline_or_transformer.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise ValidationError("No trainable parameters found after LoRA injection.")
    bad = [n for n in trainable_names if "lora_" not in n.lower()]
    if bad:
        raise ValidationError(
            f"Base weights are not frozen: {len(bad)} non-adapter params need grads "
            f"(e.g. {bad[:3]}). Only LoRA params should train."
        )
    return trainable_summary(pipeline_or_transformer)


def save_adapter(transformer: nn.Module, output_dir: str, meta: AdapterMeta) -> str:
    os.makedirs(output_dir, exist_ok=True)
    try:
        transformer.save_pretrained(output_dir)
    except Exception as exc:
        raise ValidationError(f"Could not save LoRA adapter to {output_dir}: {exc}") from exc
    return write_meta(output_dir, meta)


def load_adapter_into(transformer: nn.Module, adapter_dir: str) -> nn.Module:
    require_peft()
    from peft import PeftModel

    if not os.path.isdir(adapter_dir):
        raise ValidationError(f"Adapter directory not found: {adapter_dir}")
    if getattr(transformer, "peft_config", None):
        try:
            for name in list(transformer.peft_config.keys()):
                transformer.delete_adapter(name)
            transformer.load_adapter(adapter_dir, adapter_name="default")
            transformer.set_adapter("default")
            return transformer
        except Exception as exc:
            raise ValidationError(f"Could not load adapter from {adapter_dir}: {exc}") from exc
    try:
        loaded = PeftModel.from_pretrained(transformer, adapter_dir)
    except Exception as exc:
        raise ValidationError(f"Could not load adapter from {adapter_dir}: {exc}") from exc
    return loaded
