from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.integration


def _needs_integration():
    return os.environ.get("AUDIYO_RUN_INTEGRATION") == "1"


def test_integration_generate():
    if not _needs_integration():
        import pytest

        pytest.skip("Set AUDIYO_RUN_INTEGRATION=1 with HF access to run.")
    from audiyo import AudioModel

    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    result = model.generate(prompt="soft rain on a window", duration_seconds=4, seed=42, num_inference_steps=5)
    assert result.waveform.shape[0] == 2
    assert result.sample_rate == 44100


def test_integration_lora_smoke():
    if not _needs_integration():
        import pytest

        pytest.skip("Set AUDIYO_RUN_INTEGRATION=1 with HF access to run.")
    from audiyo import AudioModel

    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    report = model.finetune(
        dataset=os.environ.get("AUDIYO_TEST_DATASET", "tests/data"),
        output_dir=os.environ.get("AUDIYO_TEST_OUT", "tests/out_adapter"),
        max_steps=2,
        rank=4,
        alpha=4,
        duration_seconds=4,
        limit=1,
    )
    assert report.checks["finite_loss"] is True
    assert os.path.isfile(os.path.join(report.output_dir, "adapter", "audiyo_adapter.json"))
