from __future__ import annotations

GGUF_REPO = "TeamAudiyo/Minimax-Music3-GGUF"

GGUF_QUANTS = {
    "q3_k_m": {"file": "MiniMax-Music3-Q3_K_M.gguf", "gb": 1.18},
    "q4_k_m": {"file": "MiniMax-Music3-Q4_K_M.gguf", "gb": 1.49},
    "q5_k_m": {"file": "MiniMax-Music3-Q5_K_M.gguf", "gb": 1.78},
    "q6_k": {"file": "MiniMax-Music3-Q6_K.gguf", "gb": 2.06},
    "q8_0": {"file": "MiniMax-Music3-Q8_0.gguf", "gb": 2.65},
    "f16": {"file": "MiniMax-Music3-F16.gguf", "gb": 5.10},
}

DEFAULT_QUANT = "q4_k_m"

GGUF_PEAK_VRAM_GB = 4.5


def resolve_gguf_file(quant=None, gguf_file=None):
    if gguf_file is not None:
        for name, spec in GGUF_QUANTS.items():
            if spec["file"] == gguf_file:
                return {"quant": name, "file": gguf_file, "gb": spec["gb"]}
        return {"quant": quant or "custom", "file": gguf_file, "gb": 1.49}
    name = str(quant or DEFAULT_QUANT).lower().replace("-", "_")
    if name not in GGUF_QUANTS:
        from ..errors import ValidationError

        raise ValidationError("Unknown quant " + repr(quant) + ". Choose from " + str(sorted(GGUF_QUANTS)) + ".")
    spec = GGUF_QUANTS[name]
    return {"quant": name, "file": spec["file"], "gb": spec["gb"]}


def read_gguf_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            from ..errors import ValidationError

            raise ValidationError("Not a GGUF file: " + str(path))
        version = int.from_bytes(f.read(4), "little")
    info = {"magic": "GGUF", "version": version, "path": path}
    try:
        import gguf

        info["gguf_version"] = getattr(gguf, "__version__", "unknown")
        reader = gguf.GGUFReader(path)
        info["tensors"] = len(reader.tensors)
        info["fields"] = list(getattr(reader, "fields", {}).keys())[:16]
    except ImportError:
        info["gguf_version"] = "not-installed"
    except Exception as exc:
        info["error"] = str(exc)[:200]
    return info


def download_gguf_file(repo=GGUF_REPO, gguf_file=None, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        from ..errors import DependencyError

        raise DependencyError("GGUF download needs huggingface_hub.") from exc
    return hf_hub_download(repo_id=repo, filename=gguf_file or GGUF_QUANTS[DEFAULT_QUANT]["file"], token=token)


def estimate_gguf_peak_gb(quant=DEFAULT_QUANT):
    spec = resolve_gguf_file(quant)
    return round(float(spec["gb"]) + 2.4 + 0.6, 2)


def map_gguf_to_transformer(path, transformer):
    try:
        import gguf
    except ImportError as exc:
        from ..errors import DependencyError

        raise DependencyError("GGUF DiT mapping needs gguf>=0.10.0. Run pnpm-safe pip install gguf.") from exc
    import numpy as np
    import torch

    reader = gguf.GGUFReader(path)
    dest = dict(transformer.named_parameters())
    dest.update(dict(transformer.named_buffers()))
    copied = 0
    for tensor in reader.tensors:
        name = str(tensor.name).replace("/", ".").replace(":", ".")
        data = tensor.data
        if data is None:
            continue
        arr = np.asarray(data)
        for suffix_len in (3, 2, 1):
            suffix = ".".join(name.split(".")[-suffix_len:])
            for dname, param in dest.items():
                if dname.endswith(suffix) and tuple(param.shape) == tuple(arr.shape):
                    param.data.copy_(torch.from_numpy(arr.astype("float32")).to(param.device, param.dtype))
                    copied += 1
                    break
            else:
                continue
            break
    if copied == 0:
        from ..errors import ValidationError

        raise ValidationError("No GGUF tensors matched the transformer. Check the quant file.")
    return copied
