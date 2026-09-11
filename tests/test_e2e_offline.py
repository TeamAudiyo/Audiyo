from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn


class FakeLatentDist:
    def __init__(self, latents):
        self._latents = latents

    def sample(self):
        return self._latents


class FakeVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.calls = []

    def encode(self, waveforms):
        b = waveforms.shape[0]
        length = waveforms.shape[2] // 8
        gen = torch.Generator(device="cpu").manual_seed(0)
        lat = torch.randn(b, 4, length, generator=gen).to(waveforms.device)
        return SimpleNamespace(latent_dist=FakeLatentDist(lat))

    def decode(self, latents):
        b = latents.shape[0]
        wav = torch.zeros(b, 2, latents.shape[2] * 8).to(latents.device)
        return SimpleNamespace(sample=wav)

    def enable_slicing(self):
        self.calls.append("slicing")

    def enable_tiling(self):
        self.calls.append("tiling")


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.ModuleList([nn.Linear(4, 4)])
        self.mix = nn.Linear(4, 4)

    def forward(self, hidden, timestep, encoder_hidden_states=None, global_hidden_states=None, rotary_embedding=None, return_dict=False):
        h = hidden.permute(0, 2, 1)
        h = self.to_out[0](self.to_v(self.to_q(h) + self.to_k(h)))
        h = self.mix(h).permute(0, 2, 1)
        return (h,)


class FakePipeline:
    rotary_embed_dim = 4

    def __init__(self):
        self.vae = FakeVae()
        self.text_encoder = nn.Linear(4, 4)
        self.projection_model = nn.Linear(4, 4)
        self.transformer = FakeTransformer()
        self.calls = []

    def __call__(self, prompt=None, audio_end_in_s=None, audio_start_in_s=0.0, num_inference_steps=100, guidance_scale=7.0, negative_prompt=None, num_waveforms_per_prompt=1, eta=0.0, generator=None, output_type="pt"):
        length = int(round((float(audio_end_in_s) - float(audio_start_in_s)) * 44100))
        t = torch.arange(length, dtype=torch.float32)
        sine = (0.1 * torch.sin(t * 0.01)).unsqueeze(0).repeat(2, 1)
        audios = sine.unsqueeze(0).repeat(int(num_waveforms_per_prompt), 1, 1)
        return SimpleNamespace(audios=audios)

    def encode_prompt(self, captions, device, do_cfg):
        return torch.zeros(len(captions), 8, 16, device=device)

    def encode_duration(self, start, end, device, do_cfg, batch):
        s = torch.zeros(batch, 1, 16, device=device)
        e = torch.zeros(batch, 1, 16, device=device)
        return s, e

    def enable_model_cpu_offload(self):
        self.calls.append("model_offload")

    def enable_sequential_cpu_offload(self):
        self.calls.append("seq_offload")

    def enable_attention_slicing(self):
        self.calls.append("attn_slice")

    def disable_attention_slicing(self):
        self.calls.append("attn_full")


def make_model():
    from audiyo.model import AudioModel

    m = AudioModel.__new__(AudioModel)
    m._pipeline = FakePipeline()
    m.checkpoint = "test-checkpoint"
    m.device = "cpu"
    m.memory_mode = "balanced"
    m.dtype_name = "float32"
    m.applied_memory = None
    m.hardware = None
    m.max_duration = 47.55
    m._adapter_dir = None
    return m


def test_generate_orchestration_offline():
    model = make_model()
    result = model.generate(prompt="rain on a window", duration_seconds=2, seed=42, num_inference_steps=2)
    assert result.waveform.shape == (2, 88200)
    assert result.sample_rate == 44100
    assert bool(np.isfinite(result.waveform).all())
    assert result.peak > 0
    assert result.performance["latency_s"] >= 0
    assert result.settings["checkpoint"] == "test-checkpoint"


def test_generate_validates_before_compute():
    from audiyo.errors import ValidationError

    model = make_model()
    try:
        model.generate(prompt="x", duration_seconds=100)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_lora_trains_offline(tmp_path):
    import soundfile as sf

    from audiyo.datasets import build_dataset
    from audiyo.lora import load_adapter_into, read_adapter_meta
    from audiyo.training import run_lora_training

    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 440.0 * np.arange(sr * 2) / sr)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    sf.write(str(tmp_path / "a.wav"), stereo, sr)
    with open(str(tmp_path / "a.txt"), "w", encoding="utf-8") as f:
        f.write("a test tone")
    ds = build_dataset(str(tmp_path), duration_seconds=2.0)
    pipe = FakePipeline()
    before = [p.detach().clone() for p in pipe.transformer.parameters()]
    report = run_lora_training(
        pipe,
        ds,
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        rank=2,
        alpha=2,
        target_modules=["to_q", "to_k", "to_v"],
        learning_rate=1e-4,
        device="cpu",
        base_model_id="test-checkpoint",
        duration_seconds=2.0,
        progress=False,
    )
    assert report.checks["finite_loss"] is True
    assert report.checks["nonzero_adapter_grads"] is True
    assert report.checks["adapter_params_updated"] is True
    assert report.checks["resumable_checkpoint"] is True
    assert report.trainable_params > 0
    names = [n for n, p in pipe.transformer.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n.lower() for n in names)
    base_names = [n for n, p in pipe.transformer.named_parameters() if "lora_" not in n.lower()]
    for n, p in pipe.transformer.named_parameters():
        if "lora_" not in n.lower():
            assert p.requires_grad is False
    assert len(base_names) > 0
    meta = read_adapter_meta(str(tmp_path / "out" / "adapter"))
    assert meta["base_model"] == "test-checkpoint"
    fresh = FakeTransformer()
    loaded = load_adapter_into(fresh, str(tmp_path / "out" / "adapter"))
    assert loaded is not None
    moved = sum(float((a - b).abs().sum()) for a, b in zip([p.detach() for p in pipe.transformer.parameters() if p.requires_grad], [p.detach() for p in loaded.parameters() if p.requires_grad]))
    assert moved == 0.0
    _ = before


def test_resume_guard(tmp_path):
    from audiyo.errors import ValidationError
    from audiyo.training import run_lora_training

    from test_e2e_offline import FakePipeline
    import torch
    import numpy as np

    from audiyo.datasets import AudioCaptionDataset, ClipItem

    wav = np.zeros((2, 88200), dtype=np.float32)
    ds = AudioCaptionDataset([ClipItem(audio_path=None, waveform=wav, caption="x", seconds_total=2.0)], 88200, 2.0, True)
    pipe = FakePipeline()
    run_lora_training(
        pipe, ds, output_dir=str(tmp_path / "out"), max_steps=1, rank=2, alpha=2,
        target_modules=["to_q", "to_k", "to_v"], device="cpu", progress=False,
    )
    try:
        run_lora_training(
            FakePipeline(), ds, output_dir=str(tmp_path / "out"), max_steps=1, rank=2, alpha=2,
            target_modules=["to_q", "to_k", "to_v"], device="cpu", progress=False,
        )
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_preset_call_matrix():
    from audiyo.memory import apply_memory_preset

    seen = {}
    vae_seen = {}
    for mode in ["performance", "balanced", "low", "minimal"]:
        pipe = FakePipeline()
        cfg = apply_memory_preset(pipe, mode, device="cuda", dtype="float16", bf16_supported=False)
        seen[mode] = sorted(pipe.calls)
        vae_seen[mode] = sorted(pipe.vae.calls)
        assert cfg.mode == mode
    assert "model_offload" in seen["balanced"]
    assert "seq_offload" in seen["low"]
    assert "attn_slice" in seen["minimal"]
    assert "attn_full" in seen["balanced"]
    assert seen["performance"] == ["attn_full"]
    assert "slicing" in vae_seen["low"]
    assert "tiling" in vae_seen["minimal"]
    assert vae_seen["balanced"] == []
