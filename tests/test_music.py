from __future__ import annotations

import torch


def test_registry_lists_three_backends():
    from audiyo.backends import MODEL_REGISTRY

    assert set(MODEL_REGISTRY) == {"stable-audio", "testkit", "minimax-music"}


def test_detect_routes_music_checkpoint():
    from audiyo.backends import detect_model_type

    assert detect_model_type("MiniMaxAI/MiniMax-Music3") == "minimax-music"


def test_music_backend_describes_verified_facts():
    from audiyo.backends import describe_backend

    info = describe_backend("MiniMaxAI/MiniMax-Music3")
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2
    assert info["supports_negative_prompt"] is False
    assert info["training_objective"] == "unverified"
    assert "Community License" in info["license_note"]


def test_music_load_fails_clearly_without_download():
    from audiyo.backends import load_pipeline
    from audiyo.errors import AudiyoError

    try:
        load_pipeline("MiniMaxAI/MiniMax-Music3")
    except AudiyoError as exc:
        assert "22 GB" in str(exc)
        return
    raise AssertionError("expected AudiyoError")


def test_lyrics_tags_need_own_line():
    from audiyo.testkit import parse_lyrics_sections

    sections = parse_lyrics_sections("[verse]\nMorning light\n[chorus]\nSoftly breathe")
    assert [t for t, _ in sections] == ["verse", "chorus"]
    assert sections[0][1] == ["Morning light"]


def test_lyrics_text_on_tag_line_is_dropped():
    from audiyo.testkit import parse_lyrics_sections

    sections = parse_lyrics_sections("[verse] dropped words\nKept line")
    assert sections == [("verse", ["Kept line"])]


def test_music_proxy_generates_stereo():
    from audiyo.testkit import TinyMusicPipeline

    pipe = TinyMusicPipeline()
    gen = torch.Generator(device="cpu").manual_seed(3)
    out = pipe(prompt="acoustic pop", lyrics="[verse]\nMorning light", audio_duration=4, generator=gen, output_type="pt")
    wav = out.audios[0]
    assert tuple(wav.shape) == (2, 176400)
    assert out.sampling_rate == 44100
    assert bool(torch.isfinite(wav).all())


def test_music_proxy_deterministic():
    from audiyo.testkit import TinyMusicPipeline

    first = TinyMusicPipeline()(prompt="pop", lyrics="[verse]\nHi", audio_duration=2, output_type="np").audios
    second = TinyMusicPipeline()(prompt="pop", lyrics="[verse]\nHi", audio_duration=2, output_type="np").audios
    assert bool((first == second).all())
