from __future__ import annotations

import torch


def cuda_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None, "max_allocated_gb": None}
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "allocated_gb": round(float(allocated), 3),
        "reserved_gb": round(float(reserved), 3),
        "max_allocated_gb": round(float(peak), 3),
    }


def system_snapshot() -> dict:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "used_gb": round((mem.total - mem.available) / (1024 ** 3), 3),
            "total_gb": round(mem.total / (1024 ** 3), 3),
        }
    except Exception:
        return {"used_gb": None, "total_gb": None}


def reset_peak():
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
