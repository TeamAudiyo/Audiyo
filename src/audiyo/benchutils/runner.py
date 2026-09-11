from __future__ import annotations


def mean_of(values: list) -> float:
    return sum(values) / max(len(values), 1)


def summarize_latencies(latencies: list, duration_seconds: float) -> dict:
    avg = mean_of(latencies)
    return {
        "latency_mean_s": round(avg, 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
        "audio_per_wall_s": round(float(duration_seconds) / max(avg, 1e-6), 3),
    }
