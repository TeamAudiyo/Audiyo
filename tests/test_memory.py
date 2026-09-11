from __future__ import annotations


def test_memory_preset_names():
    from audiyo.memory import describe_presets

    presets = describe_presets()
    assert set(presets) == {"performance", "balanced", "low", "minimal"}


def test_memory_rejects_unknown():
    from audiyo.errors import ValidationError

    class FakePipe:
        pass

    from audiyo.memory import apply_memory_preset

    try:
        apply_memory_preset(FakePipe(), "ultra", device="cpu")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_memopt_presets_match_memory():
    from audiyo.memory import describe_presets
    from audiyo.memopt import describe_presets as describe_new

    assert set(describe_presets()) == set(describe_new())


def test_dtype_helper_rejects_unknown():
    from audiyo.errors import ValidationError
    from audiyo.utils import dtype_object

    try:
        dtype_object("int4")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")
