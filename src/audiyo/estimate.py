from __future__ import annotations

from .backends.music_stages import estimate_plan
from .config import MUSIC3_CHECKPOINT, SUPPORTED_CHECKPOINT, SUPPORTED_CHECKPOINTS, is_gguf_checkpoint
from .errors import ValidationError
from .memopt import describe_presets

STABLE_AUDIO_PEAK_GB = {
    "performance": 12.1,
    "balanced": 5.86,
    "low": 4.2,
    "minimal": 3.1,
}

STABLE_AUDIO_REST_GB = {
    "performance": 12.1,
    "balanced": 0.32,
    "low": 0.25,
    "minimal": 0.2,
}


def _is_music_checkpoint(checkpoint: str) -> bool:
    if checkpoint == MUSIC3_CHECKPOINT:
        return True
    return is_gguf_checkpoint(checkpoint)


def estimate_requirements(checkpoint: str = SUPPORTED_CHECKPOINT, memory_mode: str = "balanced", gpu_vram_gb: float | None = None) -> dict:
    if checkpoint not in SUPPORTED_CHECKPOINTS:
        raise ValidationError("unsupported checkpoint " + repr(checkpoint) + ".")
    presets = describe_presets()
    if memory_mode not in presets:
        raise ValidationError("unknown memory_mode " + repr(memory_mode) + ".")
    if _is_music_checkpoint(checkpoint):
        if is_gguf_checkpoint(checkpoint):
            from .backends.mm3_gguf import gguf_breakdown

            breakdown = gguf_breakdown()
            peak = float(breakdown["total_vram_gb"][1])
            detail = {
                "backend": "minimax-music",
                "checkpoint": checkpoint,
                "memory_mode": memory_mode,
                "peak_gb": peak,
                "breakdown_gb": breakdown,
                "resident": "one stage at a time" if memory_mode != "performance" else "all stages",
                "source": "measured figure for the GGUF path on a 15 GB card",
                "note": "Needs 9 to 10 GB free VRAM and 14 to 16 GB system RAM. Real use moves with length and steps.",
            }
        else:
            plan = estimate_plan("bfloat16")
            if memory_mode == "low":
                plan = estimate_plan("int8")
            if memory_mode == "minimal":
                plan = estimate_plan("int4")
            peak = float(plan["peak_gb"])
            detail = {
                "backend": "minimax-music",
                "checkpoint": checkpoint,
                "memory_mode": memory_mode,
                "peak_gb": peak,
                "stages_gb": plan["stages"],
                "llm_dtype": plan["llm_dtype"],
                "resident": "one stage at a time" if memory_mode != "performance" else "all stages",
                "source": "computed from published parameter counts with headroom",
                "note": "Estimate only. Real use moves with length, steps, and driver behavior.",
            }
    else:
        peak = float(STABLE_AUDIO_PEAK_GB[memory_mode])
        detail = {
            "backend": "stable-audio",
            "checkpoint": checkpoint,
            "memory_mode": memory_mode,
            "peak_gb": peak,
            "resting_gb": float(STABLE_AUDIO_REST_GB[memory_mode]),
            "resident": presets[memory_mode].get("offload", ""),
            "source": "measured once on a T4 for 10 s of audio, rounded",
            "note": "Example reading, not a promise. Longer audio and more steps use more.",
        }
    if gpu_vram_gb is not None:
        try:
            vram = float(gpu_vram_gb)
        except Exception:
            raise ValidationError("gpu_vram_gb must be a number or empty.")
        if vram <= 0:
            raise ValidationError("gpu_vram_gb must be positive.")
        if vram >= peak * 1.15:
            verdict = "fits"
        elif vram >= peak:
            verdict = "tight"
        else:
            verdict = "unlikely"
        detail["gpu_vram_gb"] = round(vram, 2)
        detail["verdict"] = verdict
    return detail
