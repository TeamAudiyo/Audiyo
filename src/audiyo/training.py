from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from .errors import ValidationError


@dataclass
class TrainReport:
    output_dir: str
    steps: int
    final_loss: float
    losses: list[float] = field(default_factory=list)
    trainable_params: int = 0
    total_params: int = 0
    checks: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def v_loss_terms(latents: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    """Cosine v-prediction noising. t is uniform in [0, 1]; the paper does not name the sampler."""
    angle = t * math.pi / 2
    while angle.dim() < latents.dim():
        angle = angle.unsqueeze(-1)
    alpha = torch.cos(angle).to(latents.dtype)
    sigma = torch.sin(angle).to(latents.dtype)
    noised = latents * alpha + noise * sigma
    target = noise * alpha - latents * sigma
    return noised, target


def encode_audio_latents(pipeline, waveforms: torch.Tensor) -> torch.Tensor:
    vae = pipeline.vae
    was_training = vae.training
    vae.eval()
    with torch.no_grad():
        posterior = vae.encode(waveforms)
        latents = posterior.latent_dist.sample()
    if was_training:
        vae.train()
    return latents


def build_conditioning(pipeline, captions: list[str], seconds_total: float, device, cfg_dropout_prob: float = 0.1):
    """Same text plus timing tensors as inference. Dropout zeroes the text part on some steps (assumed 0.1)."""
    prompt_embeds = pipeline.encode_prompt(
        captions,
        device,
        False,
    )
    start_states, end_states = pipeline.encode_duration(
        0.0,
        float(seconds_total),
        device,
        False,
        len(captions),
    )
    text_audio = torch.cat([prompt_embeds, start_states, end_states], dim=1)
    audio_only = torch.cat([start_states, end_states], dim=2)
    if cfg_dropout_prob > 0 and torch.rand(()).item() < cfg_dropout_prob:
        text_audio = torch.cat(
            [torch.zeros_like(prompt_embeds), start_states, end_states], dim=1
        )
    return text_audio, audio_only


def build_rotary(pipeline, latents: torch.Tensor, audio_embeds: torch.Tensor, device):
    from diffusers.models.embeddings import get_1d_rotary_pos_embed

    dim = pipeline.rotary_embed_dim
    length = int(latents.shape[2]) + int(audio_embeds.shape[1])
    rotary = get_1d_rotary_pos_embed(
        dim,
        length,
        use_real=True,
        repeat_interleave_real=False,
    )
    if isinstance(rotary, torch.Tensor):
        return rotary.to(device)
    return tuple(r.to(device) if isinstance(r, torch.Tensor) else r for r in rotary)


def run_lora_training(
    pipeline,
    dataset,
    *,
    output_dir: str,
    max_steps: int = 20,
    rank: int = 8,
    alpha: int = 8,
    target_modules: list[str] | tuple | None = None,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    mixed_precision: str | None = None,
    seed: int = 0,
    device: str | None = None,
    base_model_id: str = "",
    duration_seconds: float = 10.0,
    cfg_dropout_prob: float = 0.1,
    progress: bool = True,
) -> TrainReport:
    from .lora import AdapterMeta, assert_only_adapter_trainable, attach_lora, save_adapter

    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transformer = pipeline.transformer
    transformer.train()
    os.makedirs(output_dir, exist_ok=True)

    for mod in (pipeline.vae, pipeline.text_encoder, pipeline.projection_model):
        try:
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
        except Exception:
            pass
    for p in transformer.parameters():
        p.requires_grad_(False)

    transformer = attach_lora(transformer, rank=rank, alpha=alpha, target_modules=target_modules)
    pipeline.transformer = transformer
    trainable, total = assert_only_adapter_trainable(transformer)

    try:
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
    except Exception:
        pass

    params = [p for p in transformer.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    ckpt_path = os.path.join(output_dir, "trainer_state.pt")
    start_step = 0
    if os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
        except Exception:
            start_step = 0
    if start_step >= max_steps:
        raise ValidationError(
            f"Found {ckpt_path} already at step {start_step}, which reaches max_steps={max_steps}. "
            "Use a fresh output_dir or raise max_steps to keep training instead of overwriting the saved adapter."
        )

    use_amp = mixed_precision in ("fp16", "bf16") and dev.type == "cuda"
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision == "fp16" and use_amp))

    losses: list[float] = []
    t0 = time.time()
    n = len(dataset)
    if n == 0:
        raise ValidationError("Training dataset is empty.")
    step = start_step
    accum = 0
    optimizer.zero_grad(set_to_none=True)

    idx = 0
    while step < max_steps:
        item = dataset[idx % n]
        idx += 1
        wav = torch.from_numpy(item.waveform).unsqueeze(0).to(dev, dtype=torch.float32)
        captions = [item.caption]
        try:
            seconds_total = float(item.seconds_total)
        except Exception:
            seconds_total = float(duration_seconds)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            latents = encode_audio_latents(pipeline, wav).to(torch.float32)
            text_audio, audio_only = build_conditioning(
                pipeline, captions, seconds_total, dev, cfg_dropout_prob
            )
            text_audio = text_audio.to(torch.float32)
            audio_only = audio_only.to(torch.float32)
            rotary = build_rotary(pipeline, latents, audio_only, dev)
            t = torch.rand((1,), device=dev)
            noise = torch.randn_like(latents)
            noised, target = v_loss_terms(latents, noise, t)
            out = transformer(
                noised,
                t,
                encoder_hidden_states=text_audio,
                global_hidden_states=audio_only,
                rotary_embedding=rotary,
                return_dict=False,
            )
            pred = out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            loss = loss / gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise ValidationError(
                f"Non finite loss at step {step}. Check data and learning rate."
            )
        if use_amp and mixed_precision == "fp16":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum += 1
        if accum % gradient_accumulation_steps == 0:
            if use_amp and mixed_precision == "fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            if use_amp and mixed_precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            losses.append(float(loss.detach().cpu() * gradient_accumulation_steps))
            try:
                torch.save({"optimizer": optimizer.state_dict(), "step": step}, ckpt_path)
            except Exception:
                pass
            if progress and (step == 1 or step % 10 == 0 or step == max_steps):
                print(f"  step {step} of {max_steps} loss {losses[-1]:.4f}")

    after = [p.detach().clone() for p in params]
    moved = sum(float((a - b).abs().sum().cpu()) for a, b in zip(after, before))
    grads_nonzero = bool(moved > 0)

    os.makedirs(output_dir, exist_ok=True)
    from .lora import DEFAULT_TARGET_MODULES

    matched = list(target_modules) if target_modules else list(DEFAULT_TARGET_MODULES)
    save_adapter(
        transformer,
        os.path.join(output_dir, "adapter"),
        AdapterMeta(base_model=base_model_id, rank=rank, alpha=alpha, target_modules=matched),
    )
    checks = {
        "finite_loss": all(math.isfinite(x) for x in losses),
        "nonzero_adapter_grads": bool(grads_nonzero),
        "base_frozen": True,
        "adapter_params_updated": bool(moved > 0),
        "adapter_saved": os.path.isfile(os.path.join(output_dir, "adapter", "audiyo_adapter.json")),
        "resumable_checkpoint": os.path.isfile(ckpt_path),
        "note": "Smoke checks for plumbing. Not evidence of audio quality.",
    }
    return TrainReport(
        output_dir=output_dir,
        steps=step,
        final_loss=float(losses[-1]) if losses else float("nan"),
        losses=losses,
        trainable_params=trainable,
        total_params=total,
        checks=checks,
        elapsed_s=time.time() - t0,
    )
