from __future__ import annotations

import time

import torch

from .benchutils import hardware_header, summarize_latencies, versions_header
from .memory import describe_presets
from .utils import cuda_snapshot, system_snapshot
from .utils.timing import sync as _sync


def _mem():
    out = {"cuda_allocated_gb": None, "cuda_reserved_gb": None, "cuda_peak_gb": None}
    snap = cuda_snapshot()
    out["cuda_allocated_gb"] = snap["allocated_gb"]
    out["cuda_reserved_gb"] = snap["reserved_gb"]
    out["cuda_peak_gb"] = snap["max_allocated_gb"]
    sys = system_snapshot()
    out["system_used_gb"] = sys["used_gb"]
    out["system_total_gb"] = sys["total_gb"]
    return out


def run_benchmark(
    model,
    prompts,
    duration_seconds: float = 10.0,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.0,
    seed: int = 42,
    warmup: int = 1,
    repeat: int = 3,
):
    ver = versions_header()
    hw = hardware_header()
    header = {
        "checkpoint": getattr(model, "checkpoint", None),
        "device": getattr(model, "device", None),
        "dtype": getattr(model, "dtype_name", None),
        "memory_mode": getattr(model, "memory_mode", None),
        "memory_detail": model.describe_memory() if hasattr(model, "describe_memory") else {},
        "torch_version": ver["torch_version"],
        "diffusers_version": ver["diffusers_version"],
        "transformers_version": ver["transformers_version"],
        "platform": ver["platform"],
        "gpu_name": hw["gpu_name"],
        "cuda_available": hw["cuda_available"],
        "presets": describe_presets(),
    }
    runs = []
    for prompt in prompts:
        for _ in range(warmup):
            model.generate(
                prompt=prompt,
                duration_seconds=duration_seconds,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        latencies = []
        for _ in range(repeat):
            _sync()
            t0 = time.perf_counter()
            result = model.generate(
                prompt=prompt,
                duration_seconds=duration_seconds,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            _sync()
            latencies.append(time.perf_counter() - t0)
        mem = _mem()
        perf = result.performance if hasattr(result, "performance") else {}
        summary = summarize_latencies(latencies, duration_seconds)
        runs.append(
            {
                "prompt": prompt,
                "duration_seconds": duration_seconds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "latency_mean_s": summary["latency_mean_s"],
                "latency_min_s": summary["latency_min_s"],
                "latency_max_s": summary["latency_max_s"],
                "audio_per_wall_s": summary["audio_per_wall_s"],
                "memory": mem,
                "peak": result.peak if hasattr(result, "peak") else None,
                "generation_perf": perf,
            }
        )
    return {"header": header, "runs": runs}
