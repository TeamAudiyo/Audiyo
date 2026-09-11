from __future__ import annotations

import torch.nn as nn

from ..errors import ValidationError

DEFAULT_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


def list_linear_modules(module: nn.Module, prefix: str = "") -> list:
    names: list = []
    for name, child in module.named_children():
        full = prefix + "." + name if prefix else name
        if isinstance(child, nn.Linear):
            names.append(full)
        names.extend(list_linear_modules(child, full))
    return names


def find_lora_targets(transformer: nn.Module, wanted=None) -> list:
    if wanted is None:
        wanted = DEFAULT_TARGET_MODULES
    linears = list_linear_modules(transformer)
    found: list = []
    for w in wanted:
        if any(n == w or n.endswith("." + w) for n in linears):
            found.append(w)
    return found


def verify_targets_or_raise(transformer: nn.Module, wanted) -> list:
    found = find_lora_targets(transformer, wanted)
    if not found:
        sample = list_linear_modules(transformer)[:20]
        raise ValidationError(
            "None of the requested LoRA target modules were found in the transformer. "
            + "Requested "
            + str(list(wanted))
            + ". First linear layers seen: "
            + str(sample)
            + "."
        )
    return found
