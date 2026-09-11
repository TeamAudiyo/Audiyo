from __future__ import annotations

import torch

from ..hardware import detect_hardware, platform_summary


def versions_header() -> dict:
    try:
        import diffusers
        import transformers

        diffusers_v = getattr(diffusers, "__version__", "unknown")
        transformers_v = getattr(transformers, "__version__", "unknown")
    except Exception:
        diffusers_v = "unknown"
        transformers_v = "unknown"
    return {
        "torch_version": torch.__version__,
        "diffusers_version": diffusers_v,
        "transformers_version": transformers_v,
        "platform": platform_summary(),
    }


def hardware_header() -> dict:
    hw = detect_hardware("auto")
    return {
        "gpu_name": hw.gpu_name,
        "cuda_available": hw.cuda_available,
        "gpu_vram_gb": hw.gpu_vram_gb,
        "cpu_count": hw.cpu_count,
        "system_ram_gb": hw.system_ram_gb,
    }
