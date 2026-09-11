from __future__ import annotations

import hashlib

import numpy as np
import torch

MUSIC_SAMPLE_RATE = 44100
MUSIC_MAX_PROXY_SECONDS = 8.0

KNOWN_TAGS = ("intro", "verse", "pre-chorus", "chorus", "post-chorus", "bridge", "instrumental", "solo", "outro")


def parse_lyrics_sections(lyrics: str) -> list:
    sections: list = []
    current_tag = "verse"
    current_lines: list = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            sections.append((current_tag, list(current_lines)))
            current_tag = line[1 : line.index("]")].strip().lower() or "verse"
            current_lines = []
            continue
        current_lines.append(line)
    sections.append((current_tag, list(current_lines)))
    return [(t, l) for t, l in sections if l]


def section_seed(prompt: str, tag: str, index: int) -> int:
    digest = hashlib.md5((prompt + "|" + tag + "|" + str(index)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class TinyMusicPipeline:
    sampling_rate = MUSIC_SAMPLE_RATE

    def __call__(self, prompt=None, lyrics=None, audio_duration=None, num_inference_steps=10, guidance_scale=1.7, generator=None, output_type="pt", **kwargs):
        from types import SimpleNamespace

        seconds = min(float(audio_duration or 4.0), MUSIC_MAX_PROXY_SECONDS)
        total = int(seconds * MUSIC_SAMPLE_RATE)
        sections = parse_lyrics_sections(lyrics if isinstance(lyrics, str) else "")
        if not sections:
            sections = [("instrumental", [])]
        base_seed = 0
        if generator is not None:
            try:
                base_seed = int(generator.initial_seed())
            except Exception:
                base_seed = 0
        out = np.zeros(total, dtype=np.float32)
        per = max(1, total // len(sections))
        for i, (tag, lines) in enumerate(sections):
            rng = np.random.default_rng((base_seed + section_seed(str(prompt), tag, i)) % (2 ** 63))
            n = per if i < len(sections) - 1 else total - per * i
            freq = 110.0 + (abs(hash(tag)) % 8) * 55.0
            t = np.arange(n, dtype=np.float32) / MUSIC_SAMPLE_RATE
            tone = 0.08 * np.sin(2 * np.pi * freq * t)
            tone = tone + 0.03 * rng.standard_normal(n).astype(np.float32)
            edge = min(2000, n // 4)
            ramp = np.ones(n, dtype=np.float32)
            ramp[:edge] = np.linspace(0, 1, edge)
            ramp[-edge:] = np.linspace(1, 0, edge)
            out[i * per : i * per + n] = (tone * ramp).astype(np.float32)
        stereo = np.stack([out, out], axis=0)
        stereo = np.expand_dims(stereo, 0)
        if output_type == "np":
            return SimpleNamespace(audios=stereo, sampling_rate=MUSIC_SAMPLE_RATE)
        return SimpleNamespace(audios=torch.from_numpy(stereo), sampling_rate=MUSIC_SAMPLE_RATE)


def write_fake_gguf(path, dim=32):
    import struct

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(b"\x00" * 64)


class TinyStructureLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 16)
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embedding(ids))


class TinyFlowDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(16)
        self.proj = torch.nn.Linear(16, 16)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(hidden))


class TinyVocoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 8)
        self.up = torch.nn.ConvTranspose1d(8, 2, 8, stride=4, padding=2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.proj(hidden).permute(0, 2, 1)
        return self.up(h)


class TinyStagedMusicPipeline:
    sampling_rate = MUSIC_SAMPLE_RATE

    def __init__(self, device: str = "cpu"):
        from ..backends.music_stages import StageOffloader

        self.structure = TinyStructureLLM()
        self.flow = TinyFlowDiT()
        self.vocoder = TinyVocoder()
        self.offloader = StageOffloader({"structure": self.structure, "flow": self.flow, "vocoder": self.vocoder}, device)

    def __call__(self, prompt=None, lyrics=None, audio_duration=None, generator=None, output_type="pt", **kwargs):
        from types import SimpleNamespace

        seconds = min(float(audio_duration or 4.0), MUSIC_MAX_PROXY_SECONDS)
        total = int(seconds * MUSIC_SAMPLE_RATE)
        frames = (total + 3) // 4
        sections = parse_lyrics_sections(lyrics if isinstance(lyrics, str) else "")
        if not sections:
            sections = [("instrumental", [])]
        with torch.inference_mode():
            with self.offloader.stage("structure"):
                ids = torch.arange(frames, dtype=torch.long).unsqueeze(0) % 64
                struct = self.structure(ids)
                gain = float(torch.tanh(struct.float().mean()))
            with self.offloader.stage("flow"):
                latents = torch.randn(1, frames, 16, generator=generator)
                flowed = self.flow(latents)
            with self.offloader.stage("vocoder"):
                wav = self.vocoder(flowed)
        track = wav[0, :, :total]
        per = max(1, total // len(sections))
        envelope = torch.ones(total)
        for i, (tag, lines) in enumerate(sections):
            n = per if i < len(sections) - 1 else total - per * i
            edge = min(2000, n // 4)
            ramp = torch.ones(n)
            ramp[:edge] = torch.linspace(0, 1, edge)
            ramp[-edge:] = torch.linspace(1, 0, edge)
            envelope[i * per : i * per + n] = ramp
        track = track * envelope.unsqueeze(0) * (0.6 + 0.4 * gain)
        track = track / max(float(track.abs().max()), 1e-6) * 0.5
        audios = track.unsqueeze(0)
        if output_type == "np":
            return SimpleNamespace(audios=audios.numpy(), sampling_rate=MUSIC_SAMPLE_RATE)
        return SimpleNamespace(audios=audios, sampling_rate=MUSIC_SAMPLE_RATE)


class TinyFFProj(torch.nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class TinyQwenBlock(torch.nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)
        self.gate_proj = torch.nn.Linear(dim, dim * 2)
        self.up_proj = torch.nn.Linear(dim, dim * 2)
        self.down_proj = torch.nn.Linear(dim * 2, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attn = self.o_proj(self.v_proj(self.q_proj(hidden) + self.k_proj(hidden)))
        gated = self.down_proj(torch.nn.functional.gelu(self.gate_proj(hidden)) * self.up_proj(hidden))
        return attn + gated


class TinyFlowBlock(torch.nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim), torch.nn.Dropout(0.0)])
        self.ff = torch.nn.Module()
        self.ff.net = torch.nn.ModuleList([TinyFFProj(dim), torch.nn.GELU(), torch.nn.Linear(dim, dim)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h = self.to_out[0](self.to_v(self.to_q(hidden) + self.to_k(hidden)))
        h = self.ff.net[2](torch.nn.functional.gelu(self.ff.net[0](h)))
        return h + hidden


class TinyDualMusic:
    def __init__(self, dim: int = 32):
        self.language_model = TinyQwenBlock(dim)
        self.transformer = TinyFlowBlock(dim)
