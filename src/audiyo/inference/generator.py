from __future__ import annotations

import torch


def make_cpu_generator(seed: int | None):
    if seed is None:
        return None
    return torch.Generator(device="cpu").manual_seed(int(seed))
