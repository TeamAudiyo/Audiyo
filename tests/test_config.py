from __future__ import annotations


def test_generation_config_ok():
    from audiyo.config import GenerationConfig

    cfg = GenerationConfig(prompt="rain on a window", duration_seconds=10, seed=1)
    cfg.validate(max_duration=47.55)


def test_generation_config_rejects_bad_duration():
    from audiyo.config import GenerationConfig
    from audiyo.errors import ValidationError

    try:
        GenerationConfig(prompt="x", duration_seconds=100).validate(max_duration=47.55)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_generation_config_rejects_empty_prompt():
    from audiyo.config import GenerationConfig
    from audiyo.errors import ValidationError

    try:
        GenerationConfig(prompt="   ", duration_seconds=5).validate()
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_checkpoint_guard():
    from audiyo.config import check_checkpoint
    from audiyo.errors import ValidationError

    try:
        check_checkpoint("someone/else")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_finetune_config_rejects_method():
    from audiyo.config import FinetuneConfig
    from audiyo.errors import ValidationError

    try:
        FinetuneConfig(dataset="d", output_dir="o", method="full").validate()
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")
