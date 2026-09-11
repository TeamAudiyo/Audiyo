from __future__ import annotations

import time

import torch


def now() -> float:
    return time.perf_counter()


def elapsed(t0: float) -> float:
    return time.perf_counter() - t0


def sync():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
