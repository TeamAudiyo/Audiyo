from __future__ import annotations

import torch

from ..errors import ValidationError

NAMES = ("float32", "float16", "bfloat16")


def dtype_names() -> tuple:
    return NAMES


def dtype_object(name: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValidationError("Unknown dtype " + repr(name) + ".")
    return mapping[name]
