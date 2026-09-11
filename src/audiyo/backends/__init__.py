from __future__ import annotations

from typing import Any

from ..errors import CheckpointError
from .base import Backend, BackendInfo
from .minimax_music import MUSIC3_CHECKPOINT, MinimaxMusicBackend
from .stable_audio import StableAudioBackend
from .testkit_backend import TESTKIT_CHECKPOINT, TestkitBackend

MODEL_REGISTRY: dict = {
    StableAudioBackend.name: StableAudioBackend,
    TestkitBackend.name: TestkitBackend,
    MinimaxMusicBackend.name: MinimaxMusicBackend,
}

__all__ = ["Backend", "BackendInfo", "MODEL_REGISTRY", "MUSIC3_CHECKPOINT", "MinimaxMusicBackend", "StableAudioBackend", "TESTKIT_CHECKPOINT", "TestkitBackend", "detect_model_type", "describe_backend", "load_pipeline"]


def detect_model_type(checkpoint: str) -> str:
    for name, cls in MODEL_REGISTRY.items():
        backend = cls()
        info = backend.info
        if info is not None and checkpoint in info.checkpoints:
            return name
    supported: list = []
    for cls in MODEL_REGISTRY.values():
        info = cls().info
        if info is not None:
            supported.extend(info.checkpoints)
    raise CheckpointError(
        "Unsupported checkpoint " + repr(checkpoint) + ". Audiyo supports " + str(sorted(supported)) + "."
    )


def load_pipeline(checkpoint: str, **kwargs: Any) -> Any:
    return MODEL_REGISTRY[detect_model_type(checkpoint)]().load(checkpoint, **kwargs)


def describe_backend(checkpoint: str) -> dict:
    return MODEL_REGISTRY[detect_model_type(checkpoint)]().describe()
