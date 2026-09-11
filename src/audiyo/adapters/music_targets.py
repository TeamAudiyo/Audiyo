from __future__ import annotations

MUSIC_ROLES = ("llm", "transformer")

MUSIC_LLM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

MUSIC_DIT_TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")

ASSUMED_NOTE = "Assumed from public Qwen3 and diffusers DiT layouts. Verified against the loaded modules at attach time, never trusted blindly."


def resolve_music_target(target: str) -> tuple:
    if target == "llm":
        return ("llm",)
    if target == "transformer":
        return ("transformer",)
    if target == "both":
        return ("llm", "transformer")
    from ..errors import ValidationError

    raise ValidationError("target must be llm, transformer, or both. Got " + repr(target) + ".")


def validate_target_for_roles(target: str, roles) -> tuple:
    wanted = resolve_music_target(target)
    missing = [r for r in wanted if r not in list(roles)]
    if missing:
        from ..errors import ValidationError

        raise ValidationError(
            "target " + repr(target) + " needs " + str(missing) + " but this backend only has " + str(sorted(roles)) + "."
        )
    return wanted


def role_module(pipeline, role: str):
    name = "language_model" if role == "llm" else "transformer"
    module = getattr(pipeline, name, None)
    if module is None:
        from ..errors import ValidationError

        raise ValidationError("Pipeline has no " + name + " for role " + repr(role) + ".")
    return module


def assumed_targets_for(role: str) -> tuple:
    if role == "llm":
        return MUSIC_LLM_TARGETS
    if role == "transformer":
        return MUSIC_DIT_TARGETS
    from ..errors import ValidationError

    raise ValidationError("Unknown role " + repr(role) + ".")
