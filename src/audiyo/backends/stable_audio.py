from __future__ import annotations

from typing import Any

from ..config import SUPPORTED_CHECKPOINT, check_checkpoint
from ..errors import CheckpointError, auth_hint, oom_hint
from .base import Backend, BackendInfo, require_backend


class StableAudioBackend(Backend):
    name = "stable-audio"
    info = BackendInfo(
        name="stable-audio",
        checkpoints=(SUPPORTED_CHECKPOINT,),
        sample_rate=44100,
        channels=2,
        description="Stable Audio Open text-to-audio diffusion in a VAE latent space.",
        supports_negative_prompt=True,
        pipeline_class="diffusers.StableAudioPipeline",
        training_objective="v-prediction",
        roles=("transformer",),
        license_note="Weights under Stability AI Community License. Adapters inherit it.",
    )

    def load(self, checkpoint: str, **kwargs: Any) -> Any:
        require_backend()
        check_checkpoint(checkpoint)
        from diffusers import StableAudioPipeline

        torch_dtype = kwargs.get("torch_dtype")
        device_map = kwargs.get("device_map")
        token = kwargs.get("token")
        load_kwargs: dict = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if token is not None:
            load_kwargs["token"] = token
        try:
            return StableAudioPipeline.from_pretrained(checkpoint, **load_kwargs)
        except Exception as exc:
            from ..errors import scrub_text

            text = scrub_text(str(exc))
            lowered = text.lower()
            if any(k in lowered for k in ("401", "403", "gated", "unauthorized", "license", "token", "login")):
                raise auth_hint(text) from exc
            if "out of memory" in lowered or "cuda" in lowered and "memory" in lowered:
                raise oom_hint("while loading the checkpoint") from exc
            raise CheckpointError(
                "Could not load checkpoint " + repr(checkpoint) + ": " + text + "\n"
                "Hint: check the id, your network connection, and available disk/RAM."
            ) from exc
