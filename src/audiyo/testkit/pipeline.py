from __future__ import annotations

import math

import torch
import torch.nn as nn

from .modules import TinyDiT, TinyProjection, TinyTextEncoder, TinyVAE

SAMPLE_RATE = 44100
HOP = 256
WINDOW_FRAMES = 352
WINDOW_SAMPLES = WINDOW_FRAMES * HOP
LATENT_DIM = 16


def count_parameters(pipeline) -> dict:
    total = 0
    parts = {}
    for name in ["vae", "text_encoder", "projection_model", "transformer"]:
        module = getattr(pipeline, name)
        n = sum(p.numel() for p in module.parameters())
        parts[name] = int(n)
        total += n
    parts["total"] = int(total)
    return parts


class TinyPipeline:
    rotary_embed_dim = 32

    def __init__(self, vae=None, text_encoder=None, projection_model=None, transformer=None):
        self.vae = vae or TinyVAE(LATENT_DIM)
        self.text_encoder = text_encoder or TinyTextEncoder()
        self.projection_model = projection_model or TinyProjection()
        self.transformer = transformer or TinyDiT(LATENT_DIM)
        self.calls = []

    def to(self, device):
        return self

    def enable_model_cpu_offload(self):
        self.calls.append("model_offload")

    def enable_sequential_cpu_offload(self):
        self.calls.append("seq_offload")

    def enable_attention_slicing(self):
        self.calls.append("attn_slice")

    def disable_attention_slicing(self):
        self.calls.append("attn_full")

    def encode_prompt(self, captions, device, do_cfg):
        ids = self.text_encoder.tokenize(list(captions)).to(device)
        hidden = self.text_encoder(ids)
        return self.projection_model.encode_text(hidden)

    def encode_duration(self, start, end, device, do_cfg, batch):
        start_t = torch.full((batch,), float(start) if not isinstance(start, list) else float(start[0])).to(device)
        end_t = torch.full((batch,), float(end) if not isinstance(end, list) else float(end[0])).to(device)
        s = self.projection_model.encode_timing(start_t).unsqueeze(1)
        e = self.projection_model.encode_timing(end_t).unsqueeze(1)
        return s, e

    def __call__(self, prompt=None, audio_end_in_s=None, audio_start_in_s=0.0, num_inference_steps=10, guidance_scale=7.0, negative_prompt=None, num_waveforms_per_prompt=1, eta=0.0, generator=None, latents=None, output_type="pt"):
        from types import SimpleNamespace

        device = torch.device("cpu")
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)
        count = len(prompts) * int(num_waveforms_per_prompt)
        repeated = [p for p in prompts for _ in range(int(num_waveforms_per_prompt))]
        if audio_end_in_s is None:
            audio_end_in_s = WINDOW_SAMPLES / SAMPLE_RATE
        if latents is None:
            if generator is None:
                noise = torch.randn(count, LATENT_DIM, WINDOW_FRAMES)
            else:
                noise = torch.randn(count, LATENT_DIM, WINDOW_FRAMES, generator=generator)
        else:
            noise = latents
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()
        self.projection_model.eval()
        prompt_embeds = self.encode_prompt(repeated, device, False)
        start_states, end_states = self.encode_duration(audio_start_in_s, audio_end_in_s, device, False, count)
        text_audio = torch.cat([prompt_embeds, start_states, end_states], dim=1)
        audio_only = torch.cat([start_states, end_states], dim=2)
        uncond = torch.cat([torch.zeros_like(prompt_embeds), start_states, end_states], dim=1)
        with torch.inference_mode():
            current = noise
            steps = int(num_inference_steps)
            for i in range(steps):
                t = 1.0 - i / steps
                t_next = 1.0 - (i + 1) / steps
                t_batch = torch.full((count,), t)
                doubled = torch.cat([current, current], dim=0)
                doubled_t = torch.cat([t_batch, t_batch], dim=0)
                doubled_enc = torch.cat([uncond, text_audio], dim=0)
                doubled_glob = torch.cat([audio_only, audio_only], dim=0)
                pred = self.transformer(doubled, doubled_t, encoder_hidden_states=doubled_enc, global_hidden_states=doubled_glob, return_dict=False)[0]
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                velocity = pred_uncond + float(guidance_scale) * (pred_cond - pred_uncond)
                alpha = math.cos(t * math.pi / 2)
                sigma = math.sin(t * math.pi / 2)
                pred_x0 = alpha * current - sigma * velocity
                eps = sigma * current + alpha * velocity
                alpha_next = math.cos(t_next * math.pi / 2)
                sigma_next = math.sin(t_next * math.pi / 2)
                current = alpha_next * pred_x0 + sigma_next * eps
            wav_full = self.vae.decode(current).sample
        s0 = int(float(audio_start_in_s) * SAMPLE_RATE)
        s1 = int(float(audio_end_in_s) * SAMPLE_RATE)
        s0 = max(0, s0)
        s1 = min(wav_full.shape[2], s1)
        audios = wav_full[:, :, s0:s1]
        return SimpleNamespace(audios=audios)


def build_tiny_model(duration_cap: float = 47.55):
    from ..model import AudioModel

    pipe = TinyPipeline()
    model = AudioModel.__new__(AudioModel)
    model._pipeline = pipe
    model.checkpoint = "audiyo-testkit-tiny"
    model.device = "cpu"
    model.memory_mode = "balanced"
    model.dtype_name = "float32"
    model.applied_memory = None
    model.hardware = None
    model.max_duration = duration_cap
    model._adapter_dir = None
    return model
