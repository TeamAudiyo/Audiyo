from __future__ import annotations

from typing import Any

from ..config import MUSIC3_CHECKPOINT
from ..errors import CheckpointError
from .base import Backend, BackendInfo
from .music_stages import estimate_plan


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
            "peak_estimates_gb": {"bfloat16": estimate_plan("bfloat16")["peak_gb"], "int8_llm": estimate_plan("int8")["peak_gb"]},
        },
    )

    def load(self, checkpoint: str, **kwargs: Any) -> Any:
        raise CheckpointError(
            "MiniMax-Music3 cannot load in this Audiyo release. It needs a diffusers build with "
            "MiniMaxMusic3ModularPipeline (newer than the pinned 0.39.0), a CUDA GPU, and about "
            "22 GB of free VRAM in bfloat16. See docs/backends.md for the full evaluation."
        )
