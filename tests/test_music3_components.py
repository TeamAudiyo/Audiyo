from __future__ import annotations

import sys
import types

import torch


def _fake_pipe_class(store, missing=()):
    class FakePipe:
        def enable_sequential_cpu_offload(self):
            store["seq"] = True

    FakePipe.transformer = torch.nn.Linear(4, 4)
    FakePipe.vocoder = torch.nn.Linear(4, 4)
    FakePipe.language_model = None if "language_model" in missing else torch.nn.Linear(4, 4)
    FakePipe.tokenizer = None if "tokenizer" in missing else torch.nn.Linear(4, 4)
    FakePipe.rvq_depth_decoder = None if "rvq_depth_decoder" in missing else torch.nn.Linear(4, 4)
    return FakePipe


def _install_fake_stack(pipe_cls, tokenizer_ok=True, fail_tokenizer=False):
    made = {}

    def fake_tokenizer_from_pretrained(repo, **kw):
        made["tokenizer_repo"] = repo
        made["tokenizer_subfolder"] = kw.get("subfolder")
        if fail_tokenizer:
            raise RuntimeError("no network in test")
        return object()

    def fake_lm_from_pretrained(repo, **kw):
        made["lm_repo"] = repo
        made["lm_subfolder"] = kw.get("subfolder")
        return object()

    def fake_rvq_from_pretrained(repo, **kw):
        made["rvq_repo"] = repo
        made["rvq_subfolder"] = kw.get("subfolder")
        return object()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=staticmethod(fake_tokenizer_from_pretrained))
    fake_transformers.Qwen3ForCausalLM = types.SimpleNamespace(from_pretrained=staticmethod(fake_lm_from_pretrained))
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.MiniMaxMusic3ModularPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(lambda checkpoint, **kw: pipe_cls())
    )
    fake_diffusers.MiniMaxMusic3RVQDepthDecoder = types.SimpleNamespace(
        from_pretrained=staticmethod(fake_rvq_from_pretrained)
    )
    sys.modules["transformers"] = fake_transformers
    sys.modules["diffusers"] = fake_diffusers
    return made


def _clear_fake_stack():
    sys.modules.pop("transformers", None)
    sys.modules.pop("diffusers", None)


def test_full_pipe_needs_no_fallback():
    import audiyo.backends.mm3_gguf as _gguf_mod
    from audiyo.backends.minimax_music import MinimaxMusicBackend

    store = {}
    _install_fake_stack(_fake_pipe_class(store))
    real_download = _gguf_mod.download_gguf_file
    _gguf_mod.download_gguf_file = lambda repo, filename, token=None: "C:/fake/" + filename
    try:
        pipe = MinimaxMusicBackend().load("TeamAudiyo/Minimax-Music3-GGUF")
    finally:
        _gguf_mod.download_gguf_file = real_download
        _clear_fake_stack()
    assert store.get("seq") is True
    assert pipe._audiyo_fallback_components == []
    assert pipe._audiyo_gguf_file == "MiniMax-Music3-Q4_K_M.gguf"


def test_missing_front_end_is_filled_from_base():
    from audiyo.backends.minimax_music import MinimaxMusicBackend

    store = {}
    made = _install_fake_stack(_fake_pipe_class(store, missing=("tokenizer", "rvq_depth_decoder")))
    try:
        pipe = MinimaxMusicBackend().load("MiniMaxAI/MiniMax-Music3")
    finally:
        _clear_fake_stack()
    assert pipe.tokenizer is not None
    assert pipe.rvq_depth_decoder is not None
    assert sorted(pipe._audiyo_fallback_components) == ["rvq_depth_decoder", "tokenizer"]
    assert made["tokenizer_repo"] == "MiniMaxAI/MiniMax-Music3"
    assert made["tokenizer_subfolder"] == "tokenizer"
    assert made["rvq_repo"] == "MiniMaxAI/MiniMax-Music3"
    assert made["rvq_subfolder"] == "rvq_depth_decoder"


def test_fallback_failure_names_component():
    from audiyo.backends.minimax_music import MinimaxMusicBackend
    from audiyo.errors import CheckpointError

    store = {}
    _install_fake_stack(_fake_pipe_class(store, missing=("tokenizer",)), fail_tokenizer=True)
    try:
        MinimaxMusicBackend().load("MiniMaxAI/MiniMax-Music3")
    except CheckpointError as exc:
        assert "tokenizer" in str(exc)
        return
    finally:
        _clear_fake_stack()
    raise AssertionError("expected CheckpointError")


def test_quant_flags_never_reach_pipeline():
    from audiyo.backends.minimax_music import MinimaxMusicBackend

    captured_pipe = {}
    captured_lm = {}
    marker = object()

    class FakePipe:
        def enable_sequential_cpu_offload(self):
            pass

    FakePipe.transformer = torch.nn.Linear(4, 4)
    FakePipe.vocoder = torch.nn.Linear(4, 4)
    FakePipe.language_model = torch.nn.Linear(4, 4)
    FakePipe.tokenizer = torch.nn.Linear(4, 4)
    FakePipe.rvq_depth_decoder = torch.nn.Linear(4, 4)

    def fake_pipeline_from_pretrained(checkpoint, **kw):
        captured_pipe.update(kw)
        return FakePipe()

    def fake_lm_from_pretrained(repo, **kw):
        captured_lm.update(kw)
        captured_lm["repo"] = repo
        return marker

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3ForCausalLM = types.SimpleNamespace(
        from_pretrained=staticmethod(fake_lm_from_pretrained)
    )
    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.MiniMaxMusic3ModularPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(fake_pipeline_from_pretrained)
    )
    sys.modules["transformers"] = fake_transformers
    sys.modules["diffusers"] = fake_diffusers
    sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")
    try:
        pipe = MinimaxMusicBackend().load("MiniMaxAI/MiniMax-Music3", llm_quant="int8")
    finally:
        _clear_fake_stack()
        sys.modules.pop("bitsandbytes", None)
    assert "load_in_8bit" not in captured_pipe
    assert "load_in_4bit" not in captured_pipe
    assert captured_lm["load_in_8bit"] is True
    assert captured_lm["subfolder"] == "language_model"
    assert captured_lm["device_map"] == "auto"
    assert captured_lm["repo"] == "MiniMaxAI/MiniMax-Music3"
    assert pipe.language_model is marker
    assert pipe._audiyo_quantized_components == ["language_model"]


def test_balanced_sends_no_quant_config_to_pipeline():
    from audiyo.backends.minimax_music import MinimaxMusicBackend

    captured_pipe = {}
    store = {}

    def fake_pipeline_from_pretrained(checkpoint, **kw):
        captured_pipe.update(kw)
        return _fake_pipe_class(store)()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.MiniMaxMusic3ModularPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(fake_pipeline_from_pretrained)
    )
    sys.modules["diffusers"] = fake_diffusers
    try:
        pipe = MinimaxMusicBackend().load("MiniMaxAI/MiniMax-Music3", memory_mode="balanced")
    finally:
        _clear_fake_stack()
    assert "load_in_8bit" not in captured_pipe
    assert "load_in_4bit" not in captured_pipe
    assert pipe._audiyo_quantized_components == []
    assert pipe._audiyo_fallback_components == []


def test_music_call_kwargs_shape():
    from audiyo.errors import ValidationError
    from audiyo.inference import build_music_call_kwargs

    kwargs = build_music_call_kwargs("bright pop", "[verse]\nHey", 30.0, 30, None)
    assert set(kwargs) == {"prompt", "lyrics", "audio_duration", "num_inference_steps", "generator", "output_type"}
    assert kwargs["audio_duration"] == 30.0
    try:
        build_music_call_kwargs("bright pop", "   ", 30.0, 30, None)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_music_limits_reject_window_and_stacking():
    from audiyo.errors import ValidationError
    from audiyo.inference import check_music_limits

    check_music_limits(0.0, 1)
    for bad in ({"start": 2.0, "count": 1}, {"start": 0.0, "count": 2}):
        try:
            check_music_limits(bad["start"], bad["count"])
        except ValidationError:
            continue
        raise AssertionError("expected ValidationError")


def test_stable_call_kwargs_keep_window_and_guidance():
    from audiyo.config import GenerationConfig
    from audiyo.inference import build_stable_call_kwargs

    cfg = GenerationConfig(prompt="rain", duration_seconds=10, guidance_scale=7.0)
    kwargs = build_stable_call_kwargs(cfg, None)
    assert kwargs["audio_end_in_s"] == 10.0
    assert kwargs["audio_start_in_s"] == 0.0
    assert kwargs["guidance_scale"] == 7.0


def test_pipeline_max_duration_music():
    from audiyo.backend import pipeline_max_duration

    class FakeMiniMaxMusic3ModularPipeline:
        pass

    assert pipeline_max_duration(FakeMiniMaxMusic3ModularPipeline()) == 360.0
    assert pipeline_max_duration(object()) == 47.55


def test_generate_routes_music_args():
    from audiyo.model import AudioModel

    seen = {}

    class FakeOut:
        def __init__(self, audios):
            self.audios = audios

    class FakeMusicPipe:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return FakeOut(torch.zeros(1, 2, 44100, dtype=torch.float32))

    model = AudioModel()
    model._pipeline = FakeMusicPipe()
    model.checkpoint = "MiniMaxAI/MiniMax-Music3"
    model.device = "cpu"
    model.memory_mode = "balanced"
    model.dtype_name = "float32"
    model.max_duration = 360.0
    result = model.generate(
        prompt="bright pop",
        lyrics="[verse]\nHey",
        duration_seconds=2.0,
        seed=3,
        num_inference_steps=2,
    )
    assert set(seen) == {"prompt", "lyrics", "audio_duration", "num_inference_steps", "generator", "output_type"}
    assert result.settings["backend"] == "minimax-music"
    assert result.waveform.shape == (2, 44100)
