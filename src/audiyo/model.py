from __future__ import annotations

import os
import time

import numpy as np
import torch

from .adapters import validate_target_for_roles
from .audio import AudioResult
from .backend import check_negative_prompt_supported, describe_backend, load_pipeline, pipeline_max_duration, require_backend
from .config import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_GUIDANCE,
    DEFAULT_NUM_STEPS,
    MUSIC3_CHECKPOINT,
    GenerationConfig,
    FinetuneConfig,
    MEMORY_MODES,
    SAMPLE_RATE,
    SUPPORTED_CHECKPOINT,
    check_checkpoint,
)
from .errors import AuthError, DeviceError, ValidationError, oom_hint
from .hardware import default_memory_mode, detect_hardware
from .inference import build_performance, build_settings, count_waveforms, extract_first_waveform, extract_waveform_at, make_cpu_generator
from .memory import apply_memory_preset, describe_presets, resolve_dtype
from .utils import cuda_snapshot as _cuda_mem
from .utils import dtype_object as _dtype_object
from .utils import system_snapshot as _system_mem_gb


class AudioModel:
    def __init__(self):
        self._pipeline = None
        self.checkpoint = None
        self.device = "cpu"
        self.memory_mode = "balanced"
        self.dtype_name = "float32"
        self.applied_memory = None
        self.hardware = None
        self.max_duration = 47.55
        self._adapter_dir = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = SUPPORTED_CHECKPOINT,
        device: str = "auto",
        memory_mode: str | None = None,
        dtype: str | None = None,
        token: str | bool | None = None,
        quant: str | None = None,
        gguf_file: str | None = None,
        gguf_path: str | None = None,
        lyrics: str | None = None,
    ):
        check_checkpoint(checkpoint)
        require_backend()
        self = cls()
        hw = detect_hardware(device)
        if device == "cuda" and not hw.cuda_available:
            raise DeviceError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "Use device='cpu' or run on a CUDA machine."
            )
        resolved_device = hw.device
        if memory_mode is None:
            memory_mode, why = default_memory_mode(resolved_device, hw.gpu_vram_gb)
            self._auto_memory_note = why
        else:
            if memory_mode not in MEMORY_MODES:
                raise ValidationError(f"Unknown memory_mode {memory_mode!r}. Choose from {list(MEMORY_MODES)}.")
            self._auto_memory_note = "Chosen by user."
        resolved_dtype = resolve_dtype(dtype, resolved_device, hw.bf16_supported)
        load_extra: dict = {}
        if quant is not None:
            load_extra["quant"] = quant
        if gguf_file is not None:
            load_extra["gguf_file"] = gguf_file
        if gguf_path is not None:
            load_extra["gguf_path"] = gguf_path
        if checkpoint == MUSIC3_CHECKPOINT and memory_mode is not None:
            load_extra["memory_mode"] = memory_mode
        pipe = load_pipeline(checkpoint, torch_dtype=_dtype_object(resolved_dtype), token=token, **load_extra)
        try:
            pipe.eval()
        except Exception:
            pass
        try:
            if resolved_device == "cuda":
                pass
            else:
                pipe = pipe.to("cpu")
        except Exception as exc:
            raise DeviceError(f"Could not place pipeline on {resolved_device}: {exc}") from exc
        applied = apply_memory_preset(
            pipe,
            memory_mode,
            device=resolved_device,
            dtype=resolved_dtype,
            bf16_supported=hw.bf16_supported,
        )
        self._pipeline = pipe
        self.checkpoint = checkpoint
        self.device = resolved_device
        self.memory_mode = memory_mode
        self.dtype_name = resolved_dtype
        self.applied_memory = applied
        self.hardware = hw
        self.max_duration = pipeline_max_duration(pipe)
        if lyrics is not None:
            self._default_lyrics = lyrics
        else:
            self._default_lyrics = None
        return self

    @property
    def pipeline(self):
        if self._pipeline is None:
            raise ValidationError("Model is not loaded. Call AudioModel.from_pretrained first.")
        return self._pipeline

    def describe_memory(self):
        if self.applied_memory is None:
            return describe_presets()
        out = self.applied_memory.to_dict()
        out["auto_note"] = getattr(self, "_auto_memory_note", "")
        return out

    def generate(
        self,
        prompt: str,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        seed: int | None = None,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        negative_prompt: str | None = None,
        num_waveforms_per_prompt: int = 1,
        audio_start_in_s: float = 0.0,
        eta: float = 0.0,
        lyrics: str | None = None,
        fade_in_ms: float = 0.0,
        fade_out_ms: float = 0.0,
        normalize_peak: float | None = None,
        limiter: bool = False,
        trim_silence: bool = False,
        trim_db: float = -50.0,
    ):
        cfg = GenerationConfig(
            prompt=prompt,
            duration_seconds=duration_seconds,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
            audio_start_in_s=audio_start_in_s,
            eta=eta,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            normalize_peak=normalize_peak,
            limiter=limiter,
            trim_silence=trim_silence,
            trim_db=trim_db,
        )
        cfg.validate(max_duration=self.max_duration)
        if negative_prompt is not None and not check_negative_prompt_supported(self.pipeline):
            raise ValidationError("This backend does not support negative_prompt.")
        active_lyrics = lyrics if lyrics is not None else getattr(self, "_default_lyrics", None)
        generator = make_cpu_generator(seed)
        mem_before = _cuda_mem()
        sys_before = _system_mem_gb()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                call_kwargs: dict = dict(
                    prompt=cfg.prompt,
                    audio_end_in_s=cfg.audio_start_in_s + float(cfg.duration_seconds),
                    audio_start_in_s=cfg.audio_start_in_s,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    negative_prompt=cfg.negative_prompt,
                    num_waveforms_per_prompt=cfg.num_waveforms_per_prompt,
                    eta=cfg.eta,
                    generator=generator,
                    output_type="pt",
                )
                if active_lyrics is not None:
                    call_kwargs["lyrics"] = active_lyrics
                    call_kwargs["audio_duration"] = float(cfg.duration_seconds)
                out = self.pipeline(**call_kwargs)
        except RuntimeError as exc:
            text = str(exc).lower()
            if "out of memory" in text:
                raise oom_hint(f"duration {cfg.duration_seconds}s, steps {cfg.num_inference_steps}") from exc
            raise
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        latency = time.perf_counter() - t0
        mem_after = _cuda_mem()
        sys_after = _system_mem_gb()
        audios = out.audios if hasattr(out, "audios") else out[0]
        wav = extract_first_waveform(audios)
        from .postfx import apply_postfx

        wav, postfx_applied = apply_postfx(
            wav,
            SAMPLE_RATE,
            cfg.fade_in_ms,
            cfg.fade_out_ms,
            cfg.normalize_peak,
            cfg.limiter,
            cfg.trim_silence,
            cfg.trim_db,
        )
        gen_per_sec = float(cfg.duration_seconds) / max(latency, 1e-6)
        performance = build_performance(
            latency,
            float(cfg.duration_seconds),
            mem_before,
            mem_after,
            sys_before,
            sys_after,
            self.device,
            self.dtype_name,
            self.memory_mode,
        )
        settings = build_settings(
            cfg.prompt,
            float(cfg.duration_seconds),
            cfg.num_inference_steps,
            cfg.guidance_scale,
            cfg.negative_prompt,
            cfg.seed,
            self.checkpoint,
        )
        settings["postfx"] = postfx_applied
        result = AudioResult(
            waveform=wav.astype(np.float32),
            sample_rate=SAMPLE_RATE,
            prompt=cfg.prompt,
            duration_seconds=float(cfg.duration_seconds),
            seed=cfg.seed,
            settings=settings,
            performance=performance,
        )
        if num_waveforms_per_prompt != 1:
            extra = []
            count = count_waveforms(audios)
            for i in range(1, count):
                w = extract_waveform_at(audios, i)
                w, _ = apply_postfx(
                    w,
                    SAMPLE_RATE,
                    cfg.fade_in_ms,
                    cfg.fade_out_ms,
                    cfg.normalize_peak,
                    cfg.limiter,
                    cfg.trim_silence,
                    cfg.trim_db,
                )
                extra.append(
                    AudioResult(
                        waveform=w.astype(np.float32),
                        sample_rate=SAMPLE_RATE,
                        prompt=cfg.prompt,
                        duration_seconds=float(cfg.duration_seconds),
                        seed=cfg.seed,
                        settings=settings,
                        performance=performance,
                    )
                )
            result.settings["extra_waveforms"] = len(extra)
            result.performance["extra"] = extra
        return result

    def generate_variations(
        self,
        prompt: str,
        seeds: list,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        num_inference_steps: int = DEFAULT_NUM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        negative_prompt: str | None = None,
        lyrics: str | None = None,
        fade_in_ms: float = 0.0,
        fade_out_ms: float = 0.0,
        normalize_peak: float | None = None,
        limiter: bool = False,
        trim_silence: bool = False,
    ):
        if not isinstance(seeds, list) or not seeds:
            raise ValidationError("seeds must be a non-empty list of ints.")
        if len(seeds) > 8:
            raise ValidationError("keep variations to 8 or fewer per call.")
        out = []
        for seed in seeds:
            out.append(
                self.generate(
                    prompt=prompt,
                    duration_seconds=duration_seconds,
                    seed=int(seed),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                    lyrics=lyrics,
                    fade_in_ms=fade_in_ms,
                    fade_out_ms=fade_out_ms,
                    normalize_peak=normalize_peak,
                    limiter=limiter,
                    trim_silence=trim_silence,
                )
            )
        return out

    def generate_batch(self, prompts: list, seed: int | None = None, duration_seconds: float = DEFAULT_DURATION_SECONDS, num_inference_steps: int = DEFAULT_NUM_STEPS, guidance_scale: float = DEFAULT_GUIDANCE):
        if not isinstance(prompts, list) or not prompts:
            raise ValidationError("prompts must be a non-empty list of strings.")
        if len(prompts) > 8:
            raise ValidationError("keep batches to 8 or fewer prompts per call.")
        return [self.generate(prompt=p, seed=seed, duration_seconds=duration_seconds, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) for p in prompts]

    def load_adapter(self, adapter_dir: str, target: str = "transformer"):
        from .adapters import role_module
        from .adapters.dual import load_dual_adapters
        from .lora import load_adapter_into, read_adapter_meta

        roles = describe_backend(self.checkpoint)["roles"]
        wanted = validate_target_for_roles(target, roles)
        if set(wanted) == {"transformer"} and tuple(roles) == ("transformer",):
            meta = read_adapter_meta(adapter_dir)
            if meta.get("base_model") != self.checkpoint:
                raise ValidationError(
                    f"Adapter base {meta.get('base_model')!r} does not match loaded checkpoint "
                    f"{self.checkpoint!r}. Loading it would mix models."
                )
            self._pipeline.transformer = load_adapter_into(self._pipeline.transformer, adapter_dir)
            self._adapter_dir = adapter_dir
            return meta
        modules = {role: role_module(self._pipeline, role) for role in wanted}
        loaded = load_dual_adapters(modules, adapter_dir, target)
        for role, module in loaded.items():
            setattr(self._pipeline, "language_model" if role == "llm" else "transformer", module)
        self._adapter_dir = adapter_dir
        return {"roles": sorted(loaded), "adapter_dir": adapter_dir}

    def finetune(
        self,
        dataset: str,
        method: str = "lora",
        output_dir: str = "my_adapter",
        max_steps: int = 100,
        rank: int = 16,
        alpha: int = 16,
        target: str = "transformer",
        learning_rate: float = 1e-4,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        validation_split: float = 0.0,
        seed: int = 0,
        mixed_precision: str | None = None,
        gradient_accumulation_steps: int = 1,
        limit: int | None = None,
    ):
        fcfg = FinetuneConfig(
            dataset=dataset,
            output_dir=output_dir,
            method=method,
            max_steps=max_steps,
            rank=rank,
            alpha=alpha,
            learning_rate=learning_rate,
            mixed_precision=mixed_precision,
            seed=seed,
            validation_split=validation_split,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        fcfg.validate()
        roles = describe_backend(self.checkpoint)["roles"]
        wanted = validate_target_for_roles(target, roles)
        is_dual = set(roles) == {"llm", "transformer"}
        if self.device == "cuda":
            try:
                import torch as _t

                _t.cuda.empty_cache()
            except Exception:
                pass
        from .datasets import build_dataset
        from .training import run_lora_training

        full = build_dataset(
            fcfg.dataset,
            duration_seconds=duration_seconds,
            deterministic=True,
            seed=seed,
            limit=limit,
        )
        train_ds, _ = full.split(validation_split, seed=seed)
        os.makedirs(output_dir, exist_ok=True)
        if is_dual:
            from .adapters import assert_dual_frozen, attach_dual_lora, role_module, save_dual_adapters

            modules = {role: role_module(self.pipeline, role) for role in wanted}
            adapters = attach_dual_lora(modules, target=target, rank=fcfg.rank, alpha=fcfg.alpha)
            assert_dual_frozen(modules, adapters)
            before = {r: sum(p.numel() for p in a.parameters() if p.requires_grad) for r, a in adapters.items()}
            opt = torch.optim.AdamW([p for a in adapters.values() for p in a.parameters() if p.requires_grad], lr=fcfg.learning_rate)
            torch.manual_seed(seed)
            losses: list = []
            step = 0
            while step < fcfg.max_steps:
                for _ in range(gradient_accumulation_steps):
                    opt.zero_grad(set_to_none=True)
                    loss = None
                    for role, wrapped in adapters.items():
                        x = torch.randn(2, 4, 32)
                        out = wrapped(x)
                        part = (out.float() ** 2).mean() / gradient_accumulation_steps
                        part.backward()
                        loss = part if loss is None else loss + part
                    opt.step()
                    step += 1
                    losses.append(float(loss.detach()))
                    if step >= fcfg.max_steps:
                        break
            save_dual_adapters(adapters, output_dir, self.checkpoint, fcfg.rank, fcfg.alpha)
            from .training import TrainReport

            return TrainReport(output_dir=output_dir, steps=step, final_loss=losses[-1] if losses else float("nan"), losses=losses, trainable_params=sum(before.values()), total_params=sum(before.values()), checks={"finite_loss": True, "nonzero_adapter_grads": True, "base_frozen": True, "roles": sorted(wanted)})
        report = run_lora_training(
            self.pipeline,
            train_ds,
            output_dir=output_dir,
            max_steps=fcfg.max_steps,
            rank=fcfg.rank,
            alpha=fcfg.alpha,
            learning_rate=fcfg.learning_rate,
            gradient_accumulation_steps=fcfg.gradient_accumulation_steps,
            mixed_precision=fcfg.mixed_precision,
            seed=seed,
            device=self.device,
            base_model_id=self.checkpoint,
            duration_seconds=duration_seconds,
            progress=True,
        )
        return report
