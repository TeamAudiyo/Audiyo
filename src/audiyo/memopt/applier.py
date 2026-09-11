from __future__ import annotations


def apply_offload(pipeline, offload: str, device: str, notes: list) -> None:
    try:
        if offload == "model" and device == "cuda":
            pipeline.enable_model_cpu_offload()
            notes.append("Model CPU offload on: idle parts wait in RAM.")
        elif offload == "sequential" and device == "cuda":
            pipeline.enable_sequential_cpu_offload()
            notes.append("Sequential offload on: parts move one at a time, slower but leaner.")
        else:
            notes.append("No CPU offload (CPU device or performance mode).")
    except Exception as exc:
        notes.append("Offload call failed and was skipped: " + str(exc))


def apply_attention_slicing(pipeline, enabled: bool, notes: list) -> None:
    try:
        if enabled:
            pipeline.enable_attention_slicing()
            notes.append("Attention slicing on. Efficacy on this pipeline is unconfirmed.")
        else:
            try:
                pipeline.disable_attention_slicing()
            except Exception:
                pass
    except Exception as exc:
        notes.append("Attention slicing call failed: " + str(exc))


def apply_vae_flags(pipeline, slicing: bool, tiling: bool, notes: list) -> None:
    try:
        vae = getattr(pipeline, "vae", None)
        if vae is not None:
            if slicing and hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
                notes.append("VAE slicing on.")
            if tiling and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                notes.append("VAE tiling on (experimental): lower peak memory, possible seams.")
    except Exception as exc:
        notes.append("VAE slicing or tiling call failed: " + str(exc))
