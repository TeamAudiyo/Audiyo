from __future__ import annotations


def build_performance(latency_s: float, duration_s: float, mem_before: dict, mem_after: dict, sys_before: dict, sys_after: dict, device: str, dtype_name: str, memory_mode: str) -> dict:
    latency = round(float(latency_s), 3)
    per_sec = round(float(duration_s) / max(float(latency_s), 1e-6), 3)
    return {
        "latency_s": latency,
        "audio_per_wall_s": per_sec,
        "cuda_before": mem_before,
        "cuda_after": mem_after,
        "system_before_gb": sys_before,
        "system_after_gb": sys_after,
        "device": device,
        "dtype": dtype_name,
        "memory_mode": memory_mode,
    }


def build_settings(prompt: str, duration_s: float, steps: int, guidance: float, negative_prompt, seed, checkpoint) -> dict:
    return {
        "prompt": prompt,
        "duration_seconds": float(duration_s),
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "checkpoint": checkpoint,
    }
