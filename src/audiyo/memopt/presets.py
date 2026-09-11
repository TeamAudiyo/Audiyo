from __future__ import annotations

PRESET_DOCS: dict = {
    "performance": {
        "description": "Everything on GPU. Fastest, most VRAM.",
        "offload": "none",
        "attention_slicing": False,
        "vae_slicing": False,
        "vae_tiling": False,
        "tradeoff": "Needs roughly 12 plus GB VRAM at 47 s in fp16. Fails first on small cards.",
        "numerics": "Only the dtype setting changes numerics.",
    },
    "balanced": {
        "description": "Model CPU offload with full attention. Default on most machines.",
        "offload": "model",
        "attention_slicing": False,
        "vae_slicing": False,
        "vae_tiling": False,
        "tradeoff": "Small speed cost while idle components sit in RAM. VRAM drops a lot, total memory does not.",
        "numerics": "Offload itself should not change outputs. Dtype may.",
    },
    "low": {
        "description": "Sequential offload plus VAE slicing for tighter cards.",
        "offload": "sequential",
        "attention_slicing": False,
        "vae_slicing": True,
        "vae_tiling": False,
        "tradeoff": "Noticeably slower but fits 8 GB cards at moderate lengths.",
        "numerics": "Same as balanced. Slicing the VAE batch dim should be exact.",
    },
    "minimal": {
        "description": "Everything that lowers VRAM, including experimental tiling.",
        "offload": "sequential",
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "tradeoff": "Slowest. Tiled VAE decoding can cause boundary seams on long audio.",
        "numerics": "Attention slicing reorders the same math. VAE tiling may alter boundary samples slightly.",
    },
}

PRESET_NAMES = ("performance", "balanced", "low", "minimal")


def describe_presets() -> dict:
    return {k: dict(v) for k, v in PRESET_DOCS.items()}


def preset_spec(mode: str) -> dict:
    from ..errors import ValidationError

    if mode not in PRESET_DOCS:
        raise ValidationError("Unknown memory_mode " + repr(mode) + ".")
    return PRESET_DOCS[mode]
