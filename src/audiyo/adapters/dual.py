from __future__ import annotations

import os

from .meta import AdapterMeta
from .music_targets import MUSIC_ROLES, assumed_targets_for, resolve_music_target
from .targets import verify_targets_or_raise


def attach_dual_lora(modules: dict, target: str = "both", rank: int = 8, alpha: int = 16) -> dict:
    from ..lora import require_peft

    require_peft()
    from peft import LoraConfig, get_peft_model

    wanted = resolve_music_target(target)
    for module in modules.values():
        for p in module.parameters():
            p.requires_grad_(False)
    adapters: dict = {}
    for role in wanted:
        module = modules[role]
        matched = verify_targets_or_raise(module, assumed_targets_for(role))
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=list(matched),
            bias="none",
            task_type=None,
        )
        adapters[role] = get_peft_model(module, config)
    return adapters


def assert_dual_frozen(modules: dict, adapters: dict) -> dict:
    from ..errors import ValidationError

    counts = {}
    for role, wrapped in adapters.items():
        names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        if not names:
            raise ValidationError("Role " + repr(role) + " has no trainable adapter params.")
        bad = [n for n in names if "lora_" not in n.lower()]
        if bad:
            raise ValidationError("Role " + repr(role) + " left base weights trainable: " + str(bad[:3]) + ".")
        counts[role] = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    return counts


def save_dual_adapters(adapters: dict, output_dir: str, base_model: str, rank: int, alpha: int) -> str:
    import json

    from .meta import write_meta

    os.makedirs(output_dir, exist_ok=True)
    manifest_roles: list = []
    for role, wrapped in adapters.items():
        subdir = os.path.join(output_dir, role)
        try:
            wrapped.save_pretrained(subdir)
        except Exception as exc:
            from ..errors import ValidationError

            raise ValidationError("Could not save " + role + " adapter to " + subdir + ": " + str(exc)) from exc
        write_meta(subdir, AdapterMeta(base_model=base_model, rank=rank, alpha=alpha, target_modules=list(assumed_targets_for(role))))
        manifest_roles.append(role)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"base_model": base_model, "rank": rank, "alpha": alpha, "roles": manifest_roles, "known_roles": list(MUSIC_ROLES)}, f, indent=2)
    return output_dir


def load_dual_adapters(modules: dict, output_dir: str, target: str = "both") -> dict:
    from ..lora import load_adapter_into

    wanted = resolve_music_target(target)
    loaded: dict = {}
    for role in wanted:
        loaded[role] = load_adapter_into(modules[role], os.path.join(output_dir, role))
    return loaded
