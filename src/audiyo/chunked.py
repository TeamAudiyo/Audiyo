from __future__ import annotations

from typing import Any


def compare_full_vs_tiled(pipeline, latents, *, prompt_embeds=None, **kwargs) -> dict[str, Any]:
    """Decode once with tiling off and once with tiling on. Reports differences."""
    import torch

    vae = pipeline.vae
    was_tiling = bool(getattr(vae, "use_tiling", False))
    try:
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
        with torch.no_grad():
            full = vae.decode(latents).sample
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        with torch.no_grad():
            tiled = vae.decode(latents).sample
    finally:
        try:
            if was_tiling and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
            elif hasattr(vae, "disable_tiling"):
                vae.disable_tiling()
        except Exception:
            pass
    diff = (full - tiled).abs()
    result = {
        "max_abs_diff": float(diff.max().cpu()),
        "mean_abs_diff": float(diff.mean().cpu()),
        "full_shape": list(full.shape),
        "tiled_shape": list(tiled.shape),
        "note": (
            "Waveform similarity only. Listen to both before using tiled "
            "decoding for anything you keep."
        ),
    }
    return result
