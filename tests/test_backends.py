from __future__ import annotations

import torch


def test_registry_lists_three_backends():
    from audiyo.backends import MODEL_REGISTRY

    assert set(MODEL_REGISTRY) == {"stable-audio", "testkit", "minimax-music"}


def test_detect_routes_checkpoints():
    from audiyo.backends import detect_model_type

    assert detect_model_type("stabilityai/stable-audio-open-1.0") == "stable-audio"
    assert detect_model_type("audiyo-testkit-tiny") == "testkit"


def test_detect_rejects_unknown():
    from audiyo.backends import detect_model_type
    from audiyo.errors import AudiyoError

    try:
        detect_model_type("someone/else")
    except AudiyoError:
        return
    raise AssertionError("expected AudiyoError")


def test_stable_backend_describes():
    from audiyo.backends import describe_backend

    info = describe_backend("stabilityai/stable-audio-open-1.0")
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2
    assert info["supports_negative_prompt"] is True
    assert info["training_objective"] == "v-prediction"


def test_dispatcher_loads_testkit_and_generates():
    import torch

    from audiyo.backends import load_pipeline

    pipe = load_pipeline("audiyo-testkit-tiny")
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = pipe(prompt="rain", audio_end_in_s=2, num_inference_steps=2, generator=gen, output_type="pt")
    wav = out.audios[0]
    assert tuple(wav.shape) == (2, 88200)
    assert bool(torch.isfinite(wav).all())
