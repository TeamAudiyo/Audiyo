from __future__ import annotations

from .dual import assert_dual_frozen, attach_dual_lora, load_dual_adapters, save_dual_adapters
from .meta import AdapterMeta, meta_path, write_meta, read_meta
from .music_targets import ASSUMED_NOTE, MUSIC_DIT_TARGETS, MUSIC_LLM_TARGETS, MUSIC_ROLES, assumed_targets_for, resolve_music_target, role_module, validate_target_for_roles
from .targets import DEFAULT_TARGET_MODULES, list_linear_modules, find_lora_targets, verify_targets_or_raise

__all__ = ["AdapterMeta", "meta_path", "write_meta", "read_meta", "DEFAULT_TARGET_MODULES", "list_linear_modules", "find_lora_targets", "verify_targets_or_raise", "assert_dual_frozen", "attach_dual_lora", "load_dual_adapters", "save_dual_adapters", "ASSUMED_NOTE", "MUSIC_DIT_TARGETS", "MUSIC_LLM_TARGETS", "MUSIC_ROLES", "assumed_targets_for", "resolve_music_target", "role_module", "validate_target_for_roles"]
