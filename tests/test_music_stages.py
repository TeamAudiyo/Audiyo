from __future__ import annotations

import torch


def test_estimate_math_is_explicit():
    from audiyo.backends.music_stages import estimate_stage_gb

    assert estimate_stage_gb(8.6, "bfloat16", 0.2) == round(8.6 * 2.0 * 1.2, 2)
    assert estimate_stage_gb(8.6, "int8", 0.2) == round(8.6 * 1.0 * 1.2, 2)


def test_bf16_peak_exceeds_16gb_cards():
    from audiyo.backends.music_stages import estimate_plan

    plan = estimate_plan("bfloat16")
    assert plan["peak_gb"] > 16.0
    assert plan["stages"]["structure"] == plan["peak_gb"]


def test_int8_llm_fits_12gb_cards():
    from audiyo.backends.music_stages import estimate_plan

    plan = estimate_plan("int8")
    assert plan["peak_gb"] < 12.0


def test_preset_plan_covers_four_modes():
    from audiyo.backends.music_stages import PRESET_STAGE_PLAN

    assert set(PRESET_STAGE_PLAN) == {"performance", "balanced", "low", "minimal"}
    assert PRESET_STAGE_PLAN["low"]["llm_dtype"] == "int8"
    assert PRESET_STAGE_PLAN["minimal"]["llm_dtype"] == "int4"


def test_offloader_runs_stages_in_order():
    from audiyo.backends.music_stages import StageOffloader
    from audiyo.testkit.music import TinyFlowDiT, TinyStructureLLM, TinyVocoder

    off = StageOffloader({"structure": TinyStructureLLM(), "flow": TinyFlowDiT(), "vocoder": TinyVocoder()}, "cpu")
    with off.stage("structure"):
        assert off.active == "structure"
    with off.stage("flow"):
        assert off.active == "flow"
    with off.stage("vocoder"):
        assert off.active == "vocoder"
    assert off.order == ["structure", "flow", "vocoder"]


def test_offloader_rejects_bad_stage():
    from audiyo.backends.music_stages import StageOffloader
    from audiyo.errors import AudiyoError
    from audiyo.testkit.music import TinyFlowDiT, TinyStructureLLM, TinyVocoder

    off = StageOffloader({"structure": TinyStructureLLM(), "flow": TinyFlowDiT(), "vocoder": TinyVocoder()}, "cpu")
    try:
        off.activate("vocals")
    except AudiyoError:
        return
    raise AssertionError("expected AudiyoError")


def test_staged_proxy_generates_deterministically():
    from audiyo.testkit import TinyStagedMusicPipeline

    pipe = TinyStagedMusicPipeline()

    def run():
        gen = torch.Generator(device="cpu").manual_seed(9)
        return pipe(prompt="pop", lyrics="[verse]\nHi", audio_duration=2, generator=gen, output_type="np").audios

    first = run()
    second = run()
    assert first.shape == (1, 2, 88200)
    assert bool((first == second).all())
    assert bool((first != 0).any())


def test_staged_proxy_runs_all_stages():
    import torch

    from audiyo.testkit import TinyStagedMusicPipeline

    pipe = TinyStagedMusicPipeline()
    gen = torch.Generator(device="cpu").manual_seed(5)
    out = pipe(prompt="pop", lyrics="[verse]\nHi\n[chorus]\nHey", audio_duration=2, generator=gen, output_type="pt")
    assert pipe.offloader.order == ["structure", "flow", "vocoder"]
    assert tuple(out.audios.shape) == (1, 2, 88200)
    assert bool(torch.isfinite(out.audios).all())
