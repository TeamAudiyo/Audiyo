from __future__ import annotations

from typing import Any

from ..config import MM3_GGUF_FILES, MM3_GGUF_REPO, MUSIC3_CHECKPOINT, check_checkpoint
from ..errors import CheckpointError, DependencyError, auth_hint, oom_hint, scrub_text
from .base import Backend, BackendInfo, require_backend
from .mm3_gguf import DEFAULT_QUANT, GGUF_QUANTS, GGUF_REPO, estimate_gguf_peak_gb, map_gguf_to_transformer, resolve_gguf_file
from .music_stages import PRESET_STAGE_PLAN, StageOffloader, estimate_plan


class MinimaxMusicBackend(Backend):
    name = "minimax-music"
    info = BackendInfo(
        name="minimax-music",
        checkpoints=(MUSIC3_CHECKPOINT,),
        sample_rate=44100,
        channels=2,
        description="Lyrics plus description conditioned song generation with an autoregressive front end and flow-matching synthesis.",
        supports_negative_prompt=False,
        pipeline_class="diffusers.MiniMaxMusic3ModularPipeline",
        training_objective="unverified",
        roles=("llm", "transformer"),
        license_note="Weights under the MiniMax-Music3 Community License. Not Apache or MIT. Commercial use needs UI attribution and authorization above 20M dollars yearly revenue.",
        extra={
            "conditioning": "prompt plus lyrics plus audio_duration",
            "max_duration_seconds": 360,
            "components": "Qwen3ForCausalLM plus MiniMaxMusic3RVQDepthDecoder plus MiniMaxMusic3ConditionEncoder plus MiniMaxMusic3Transformer1DModel plus FlowMatchEulerDiscreteScheduler plus MiniMaxMusic3Vocoder",
            "lora_targets": "unverified",
            "stages": "structure plus flow plus vocoder, sequential with one resident",
            "peak_estimates_gb": {"bfloat16": estimate_plan("bfloat16")["peak_gb"], "int8_llm": estimate_plan("int8")["peak_gb"], "gguf_q4_k_m": estimate_gguf_peak_gb("q4_k_m")},
            "gguf_repo": GGUF_REPO,
            "gguf_quants": sorted(GGUF_QUANTS),
            "default_quant": DEFAULT_QUANT,
            "gguf_checkpoint": MM3_GGUF_REPO,
        },
    )

    def load(self, checkpoint: str, **kwargs: Any) -> Any:
        require_backend()
        check_checkpoint(checkpoint)
        gguf_request = None
        if checkpoint == MM3_GGUF_REPO:
            gguf_request = GGUF_QUANTS[DEFAULT_QUANT]["file"]
        elif checkpoint.startswith(MM3_GGUF_REPO + ":"):
            gguf_request = checkpoint.split(":", 1)[1]
            if gguf_request not in MM3_GGUF_FILES:
                from ..errors import ValidationError

                raise ValidationError("Unknown GGUF file " + repr(gguf_request) + ".")
        base_checkpoint = MUSIC3_CHECKPOINT
        try:
            from diffusers import MiniMaxMusic3ModularPipeline
        except ImportError as exc:
            raise DependencyError(
                "MiniMax-Music3 needs a diffusers build with MiniMaxMusic3ModularPipeline "
                "(newer than the pinned 0.39.0). Upgrade diffusers, then retry. Peak is about "
                + str(estimate_plan("bfloat16")["peak_gb"])
                + " GB in bfloat16 sequential, about "
                + str(estimate_plan("int8")["peak_gb"])
                + " GB with the LLM in 8-bit."
            ) from exc
        memory_mode = kwargs.pop("memory_mode", "balanced")
        if memory_mode not in PRESET_STAGE_PLAN:
            from ..errors import ValidationError

            raise ValidationError("Unknown memory_mode " + repr(memory_mode) + ".")
        plan = PRESET_STAGE_PLAN[memory_mode]
        llm_quant = kwargs.pop("llm_quant", None)
        if llm_quant is None:
            llm_quant = "none" if plan["llm_dtype"] == "bfloat16" else plan["llm_dtype"]
            try:
                import torch as _t

                if _t.cuda.is_available():
                    total = float(_t.cuda.get_device_properties(0).total_memory) / (1024 ** 3)
                    if total <= 13.0:
                        llm_quant = "int4"
                    elif total <= 21.0:
                        llm_quant = "int8"
            except Exception:
                pass
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is None:
            import torch

            torch_dtype = torch.bfloat16
        token = kwargs.pop("token", None)
        device_map = kwargs.pop("device_map", None)
        quant = kwargs.pop("quant", None)
        gguf_file = kwargs.pop("gguf_file", None)
        gguf_path = kwargs.pop("gguf_path", None)
        gguf_repo = kwargs.pop("gguf_repo", GGUF_REPO)
        use_gguf = quant is not None or gguf_file is not None or gguf_path is not None
        load_kwargs: dict = {"torch_dtype": torch_dtype}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if token is not None:
            load_kwargs["token"] = token
        want = str(llm_quant).lower()
        if want in ("int8", "int4", "8bit", "4bit"):
            try:
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise DependencyError("mode " + repr(memory_mode) + " needs bitsandbytes.") from exc
            if want in ("int8", "8bit"):
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["load_in_4bit"] = True
        try:
            pipe = MiniMaxMusic3ModularPipeline.from_pretrained(base_checkpoint, **load_kwargs, **kwargs)
        except Exception as exc:
            text = scrub_text(str(exc))
            lowered = text.lower()
            if any(k in lowered for k in ("401", "403", "gated", "unauthorized", "token", "login")):
                raise auth_hint(text) from exc
            if "out of memory" in lowered:
                raise oom_hint("loading MiniMax-Music3") from exc
            raise CheckpointError("Could not load " + repr(checkpoint) + ": " + text) from exc
        try:
            pipe.enable_sequential_cpu_offload()
            pipe._audiyo_stage_mode = "sequential-cpu-offload"
        except Exception:
            try:
                stages = {}
                pairs = (("structure", ("language_model", "structure")), ("flow", ("transformer", "flow")), ("vocoder", ("vocoder", "vae")))
                ok = True
                for stage_name, attrs in pairs:
                    module = None
                    for attr in attrs:
                        module = getattr(pipe, attr, None)
                        if module is not None:
                            break
                    if module is None:
                        ok = False
                        break
                    stages[stage_name] = module
                if ok:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    pipe._audiyo_offloader = StageOffloader(stages, device)
                    pipe._audiyo_stage_mode = "audiyo-stage-offloader:" + device
                else:
                    pipe._audiyo_stage_mode = "resident"
            except Exception:
                pipe._audiyo_stage_mode = "resident"
        pipe._audiyo_memory_mode = memory_mode
        pipe._audiyo_llm_quant = llm_quant
        if gguf_request is not None and gguf_file is None and quant is None:
            gguf_file = gguf_request
            use_gguf = True
        if use_gguf:
            spec = resolve_gguf_file(quant, gguf_file)
            if gguf_path is None:
                try:
                    from .mm3_gguf import download_gguf_file

                    gguf_path = download_gguf_file(gguf_repo, spec["file"], token)
                except Exception as exc:
                    raise CheckpointError("Could not download " + spec["file"] + ": " + scrub_text(str(exc))) from exc
            transformer = getattr(pipe, "transformer", None)
            if transformer is not None and gguf_path is not None:
                try:
                    pipe._audiyo_gguf_mapped = map_gguf_to_transformer(gguf_path, transformer)
                except Exception as exc:
                    pipe._audiyo_gguf_error = scrub_text(str(exc))[:300]
                    pipe._audiyo_gguf_mapped = 0
            pipe._audiyo_quant = spec["quant"]
            pipe._audiyo_gguf_file = spec["file"]
            pipe._audiyo_gguf_path = gguf_path
        else:
            pipe._audiyo_quant = "bf16"
        return pipe
