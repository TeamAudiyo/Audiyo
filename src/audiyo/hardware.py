from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    gpu_vram_gb: float | None
    cpu_count: int
    system_ram_gb: float | None
    bf16_supported: bool
    fp16_supported: bool
    torch_version: str
    reason: str = ""


def _system_ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        return None


def detect_hardware(device: str | None = None) -> HardwareInfo:
    import torch

    asked = device or "auto"
    cuda = torch.cuda.is_available()
    gpu_name: str | None = None
    vram: float | None = None
    if cuda:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            gpu_name = props.name
            vram = round(props.total_memory / (1024**3), 2)
        except Exception:
            pass
    bf16 = False
    try:
        bf16 = cuda and torch.cuda.is_bf16_supported()
    except Exception:
        bf16 = False
    if device is None or device == "auto":
        resolved = "cuda" if cuda else "cpu"
        reason = "CUDA GPU found." if cuda else "No CUDA GPU found; using CPU."
    elif device == "cuda":
        resolved = "cuda"
        reason = "Requested by user."
    elif device == "cpu":
        resolved = "cpu"
        reason = "Requested by user."
    else:
        from .errors import DeviceError

        raise DeviceError(
            f"Unknown device {device!r}. Use 'auto', 'cuda', or 'cpu'."
        )
    return HardwareInfo(
        device=resolved,
        cuda_available=cuda,
        gpu_name=gpu_name,
        gpu_vram_gb=vram,
        cpu_count=os.cpu_count() or 1,
        system_ram_gb=_system_ram_gb(),
        bf16_supported=bool(bf16),
        fp16_supported=True,
        torch_version=torch.__version__,
        reason=f"Asked for {asked!r}. {reason}",
    )


def default_dtype(device: str, bf16_supported: bool) -> str:
    if device == "cpu":
        return "float32"
    return "bfloat16" if bf16_supported else "float16"


def default_memory_mode(device: str, vram_gb: float | None) -> tuple[str, str]:
    if device == "cpu":
        return "balanced", "CPU has no VRAM to save; balanced keeps full precision on CPU."
    if vram_gb is None:
        return "balanced", "VRAM size unknown; balanced is the safe middle ground."
    if vram_gb >= 20:
        return "performance", f"{vram_gb} GB VRAM leaves room for the full model."
    if vram_gb >= 10:
        return "balanced", f"{vram_gb} GB VRAM fits the model with offload as a cushion."
    if vram_gb >= 6:
        return "low", f"{vram_gb} GB VRAM is tight for a 1B DiT; sequential offload helps."
    return "minimal", f"{vram_gb} GB VRAM is very tight; minimal trades speed for fit."


def platform_summary() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
