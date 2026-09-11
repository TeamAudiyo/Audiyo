from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half).to(values.device)
    args = values.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TinyAttention(nn.Module):
    def __init__(self, dim: int, cross_dim: int | None = None):
        super().__init__()
        kv_dim = cross_dim if cross_dim is not None else dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(kv_dim, dim)
        self.to_v = nn.Linear(kv_dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.scale = dim ** -0.5

    def forward(self, hidden: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = hidden if context is None else context
        q = self.to_q(hidden)
        k = self.to_k(context)
        v = self.to_v(context)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(weights, v)
        out = self.to_out[0](out)
        return self.to_out[1](out)


class TinyDiTBlock(nn.Module):
    def __init__(self, dim: int, cross_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = TinyAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = TinyAttention(dim, cross_dim=cross_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, dim * 2)
        self.ff_out = nn.Linear(dim * 2, dim)

    def forward(self, hidden: torch.Tensor, encoder: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn1(self.norm1(hidden))
        hidden = hidden + self.attn2(self.norm2(hidden), encoder)
        hidden = hidden + self.ff_out(torch.nn.functional.gelu(self.ff_in(self.norm3(hidden))))
        return hidden


class TinyDiT(nn.Module):
    def __init__(self, latent_dim: int = 16, width: int = 256, depth: int = 6, cross_dim: int = 96):
        super().__init__()
        self.in_proj = nn.Linear(latent_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.cond_pool = nn.Linear(192, width)
        self.blocks = nn.ModuleList([TinyDiTBlock(width, cross_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, latent_dim)

    def forward(self, hidden_states, timestep, encoder_hidden_states=None, global_hidden_states=None, rotary_embedding=None, return_dict=False):
        h = self.in_proj(hidden_states.permute(0, 2, 1))
        t = sinusoidal_embedding(timestep.float().flatten(), 256).to(h.dtype)
        t = self.time_mlp(t).unsqueeze(1)
        if global_hidden_states is not None:
            g = self.cond_pool(global_hidden_states.float().mean(dim=1)).unsqueeze(1).to(h.dtype)
        else:
            g = torch.zeros_like(t)
        h = h + t + g
        enc = encoder_hidden_states if encoder_hidden_states is not None else h
        for block in self.blocks:
            h = block(h, enc.float())
        out = self.out_proj(self.out_norm(h)).permute(0, 2, 1)
        return (out,)


class TinyVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_conv = nn.Conv1d(2, 32, 7, padding=3)
        self.down0 = nn.Conv1d(32, 64, 8, stride=4, padding=2)
        self.down1 = nn.Conv1d(64, 128, 8, stride=4, padding=2)
        self.down2 = nn.Conv1d(128, 128, 8, stride=4, padding=2)
        self.down3 = nn.Conv1d(128, 128, 8, stride=4, padding=2)
        self.to_stats = nn.Conv1d(128, latent_dim * 2, 3, padding=1)
        self.from_latent = nn.Conv1d(latent_dim, 128, 3, padding=1)
        self.up0 = nn.ConvTranspose1d(128, 128, 8, stride=4, padding=2)
        self.up1 = nn.ConvTranspose1d(128, 64, 8, stride=4, padding=2)
        self.up2 = nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2)
        self.up3 = nn.ConvTranspose1d(32, 32, 8, stride=4, padding=2)
        self.out_conv = nn.Conv1d(32, 2, 7, padding=3)
        self.use_slicing = False
        self.use_tiling = False
        self.tile_frames = 176
        self.tile_overlap = 8

    def encode(self, waveforms: torch.Tensor):
        from types import SimpleNamespace

        h = torch.relu(self.in_conv(waveforms))
        h = torch.relu(self.down0(h))
        h = torch.relu(self.down1(h))
        h = torch.relu(self.down2(h))
        h = torch.relu(self.down3(h))
        stats = self.to_stats(h)
        mu, logvar = stats.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)
        latents = mu + std * torch.randn_like(std)
        dist = SimpleNamespace(sample=lambda: latents, mode=lambda: mu)
        return SimpleNamespace(latent_dist=dist)

    def decode_frame(self, latents: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.from_latent(latents))
        h = torch.relu(self.up0(h))
        h = torch.relu(self.up1(h))
        h = torch.relu(self.up2(h))
        h = torch.relu(self.up3(h))
        return self.out_conv(h)

    def decode(self, latents: torch.Tensor):
        from types import SimpleNamespace

        if self.use_tiling and latents.shape[2] > self.tile_frames:
            parts = []
            half = self.tile_overlap // 2
            step = self.tile_frames - self.tile_overlap
            start = 0
            total = latents.shape[2]
            while start < total:
                end = min(start + self.tile_frames, total)
                chunk = self.decode_frame(latents[:, :, start:end])
                trim_left = half * 256 if start > 0 else 0
                trim_right = half * 256 if end < total else 0
                if trim_right > 0:
                    chunk = chunk[:, :, trim_left:-trim_right]
                else:
                    chunk = chunk[:, :, trim_left:]
                parts.append(chunk)
                if end >= total:
                    break
                start += step
            wav = torch.cat(parts, dim=2)
        else:
            wav = self.decode_frame(latents)
        return SimpleNamespace(sample=wav)

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab: int = 512, dim: int = 128, layers: int = 2, max_len: int = 16):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True) for _ in range(layers)])

    def tokenize(self, captions: list) -> torch.Tensor:
        ids = []
        for text in captions:
            digest = hashlib.md5(text.encode("utf-8")).digest()
            row = [(digest[i % len(digest)] + i * 31) % 511 + 1 for i in range(self.max_len)]
            ids.append(row)
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embedding(input_ids) + self.pos[:, : input_ids.shape[1], :]
        for layer in self.layers:
            h = layer(h)
        return h


class TinyProjection(nn.Module):
    def __init__(self, text_dim: int = 128, cross_dim: int = 96):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, cross_dim)
        self.time_proj = nn.Linear(48, cross_dim)

    def encode_text(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.text_proj(hidden)

    def encode_timing(self, seconds: torch.Tensor) -> torch.Tensor:
        return self.time_proj(sinusoidal_embedding(seconds, 48).to(self.time_proj.weight.dtype))
