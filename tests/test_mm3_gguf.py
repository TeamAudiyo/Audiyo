from __future__ import annotations

import os
import tempfile

import torch


def test_detect_routes_gguf_checkpoint():
    from audiyo.backends import detect_model_type

    assert detect_model_type("TeamAudiyo/Minimax-Music3-GGUF") == "minimax-music"
    assert detect_model_type("TeamAudiyo/Minimax-Music3-GGUF:MiniMax-Music3-Q4_K_M.gguf") == "minimax-music"


def test_gguf_checkpoint_selects_default_file():
    import sys
    import types

    import torch

    from audiyo.backends.minimax_music import MinimaxMusicBackend

    seen = {}

    class FakePipe:
        language_model = torch.nn.Linear(4, 4)
        transformer = torch.nn.Linear(4, 4)
        vocoder = torch.nn.Linear(4, 4)
        tokenizer = torch.nn.Linear(4, 4)
        rvq_depth_decoder = torch.nn.Linear(4, 4)

        def enable_sequential_cpu_offload(self):
            seen["seq"] = True

    tensor = types.SimpleNamespace(name="transformer.weight", data=None)
    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = lambda path: types.SimpleNamespace(tensors=[tensor], fields={})
    sys.modules["gguf"] = fake_gguf
    mod = types.ModuleType("diffusers")
    mod.MiniMaxMusic3ModularPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(lambda checkpoint, **kw: FakePipe())
    )
    sys.modules["diffusers"] = mod
    import audiyo.backends.mm3_gguf as _gguf_mod

    real_download = _gguf_mod.download_gguf_file
    _gguf_mod.download_gguf_file = lambda repo, filename, token=None: "C:/fake/" + filename
    try:
        pipe = MinimaxMusicBackend().load("TeamAudiyo/Minimax-Music3-GGUF")
    finally:
        _gguf_mod.download_gguf_file = real_download
        sys.modules.pop("diffusers", None)
        sys.modules.pop("gguf", None)
    assert seen.get("seq") is True
    assert pipe._audiyo_gguf_file == "MiniMax-Music3-Q4_K_M.gguf"
    assert pipe._audiyo_quant == "q4_k_m"


def test_gguf_quant_table_defaults_q4():
    from audiyo.backends.mm3_gguf import DEFAULT_QUANT, GGUF_QUANTS, resolve_gguf_file

    assert DEFAULT_QUANT == "q4_k_m"
    spec = resolve_gguf_file("q4_k_m")
    assert spec["file"] == "MiniMax-Music3-Q4_K_M.gguf"
    assert spec["gb"] == 1.49
    assert set(GGUF_QUANTS) == {"q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16"}
    try:
        resolve_gguf_file("q9")
    except Exception:
        return
    raise AssertionError("expected ValidationError")


def test_gguf_header_mock_offline():
    from audiyo.backends.mm3_gguf import read_gguf_header
    from audiyo.testkit import write_fake_gguf

    d = tempfile.mkdtemp()
    path = os.path.join(d, "fake.gguf")
    write_fake_gguf(path)
    info = read_gguf_header(path)
    assert info["magic"] == "GGUF"
    assert info["version"] == 3


def test_gguf_peak_under_45():
    from audiyo.backends.mm3_gguf import estimate_gguf_peak_gb

    assert estimate_gguf_peak_gb("q4_k_m") <= 4.5


def test_gguf_load_path_sets_quant_and_offload():
    import sys
    import types

    from audiyo.backends.minimax_music import MinimaxMusicBackend

    made = {}

    class FakePipe:
        language_model = torch.nn.Linear(4, 4)
        transformer = torch.nn.Linear(4, 4)
        vocoder = torch.nn.Linear(4, 4)
        tokenizer = torch.nn.Linear(4, 4)
        rvq_depth_decoder = torch.nn.Linear(4, 4)

        def enable_sequential_cpu_offload(self):
            made["seq"] = True

    tensor = types.SimpleNamespace(name="transformer.weight", data=None)
    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = lambda path: types.SimpleNamespace(tensors=[tensor], fields={})
    sys.modules["gguf"] = fake_gguf
    mod = types.ModuleType("diffusers")
    mod.MiniMaxMusic3ModularPipeline = types.SimpleNamespace(
        from_pretrained=staticmethod(lambda checkpoint, **kw: FakePipe())
    )
    sys.modules["diffusers"] = mod
    d = tempfile.mkdtemp()
    path = os.path.join(d, "MiniMax-Music3-Q4_K_M.gguf")
    open(path, "wb").write(b"GGUF" + b"\x00" * 64)
    try:
        pipe = MinimaxMusicBackend().load("MiniMaxAI/MiniMax-Music3", quant="q4_k_m", gguf_path=path)
    finally:
        sys.modules.pop("diffusers", None)
        sys.modules.pop("gguf", None)
    assert made.get("seq") is True
    assert pipe._audiyo_quant == "q4_k_m"
    assert pipe._audiyo_gguf_file == "MiniMax-Music3-Q4_K_M.gguf"


def test_dual_lora_grads_only_on_adapters():
    from audiyo.adapters import assert_dual_frozen, attach_dual_lora
    from audiyo.testkit import TinyDualMusic

    dual = TinyDualMusic()
    modules = {"llm": dual.language_model, "transformer": dual.transformer}
    adapters = attach_dual_lora(modules, target="both", rank=2, alpha=2)
    counts = assert_dual_frozen(modules, adapters)
    assert counts["llm"] > 0
    assert counts["transformer"] > 0
    opt = torch.optim.SGD([p for a in adapters.values() for p in a.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad()
    loss = sum((a(torch.randn(2, 4, 32)).float() ** 2).mean() for a in adapters.values())
    loss.backward()
    for role, wrapped in adapters.items():
        trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        assert trainable and all("lora_" in n.lower() for n in trainable)
    opt.step()


def test_dual_save_reload_roundtrip():
    import json

    from audiyo.adapters import attach_dual_lora, load_dual_adapters, save_dual_adapters
    from audiyo.testkit import TinyDualMusic

    d = tempfile.mkdtemp()
    out = os.path.join(d, "dual")
    dual = TinyDualMusic()
    adapters = attach_dual_lora({"llm": dual.language_model, "transformer": dual.transformer}, target="both", rank=2, alpha=2)
    save_dual_adapters(adapters, out, "MiniMaxAI/MiniMax-Music3", 2, 2)
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    assert manifest["roles"] == ["llm", "transformer"]
    fresh = TinyDualMusic()
    reloaded = load_dual_adapters({"llm": fresh.language_model, "transformer": fresh.transformer}, out, target="both")
    assert set(reloaded) == {"llm", "transformer"}
