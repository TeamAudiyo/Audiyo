from __future__ import annotations

from typing import Any

from .base import Backend, BackendInfo

TESTKIT_CHECKPOINT = "audiyo-testkit-tiny"


class TestkitBackend(Backend):
    name = "testkit"
    info = BackendInfo(
        name="testkit",
        checkpoints=(TESTKIT_CHECKPOINT,),
        sample_rate=44100,
        channels=2,
        description="Tiny stand-in with the Stable Audio layout for offline tests.",
        supports_negative_prompt=True,
        pipeline_class="audiyo.testkit.TinyPipeline",
        training_objective="v-prediction",
        roles=("transformer",),
        license_note="Bundled test code under the Audiyo license. Not a real model.",
    )

    def load(self, checkpoint: str, **kwargs: Any) -> Any:
        from ..errors import ValidationError
        from ..testkit import TinyPipeline

        if checkpoint != TESTKIT_CHECKPOINT:
            raise ValidationError("Unknown testkit checkpoint " + repr(checkpoint) + ".")
        return TinyPipeline()
