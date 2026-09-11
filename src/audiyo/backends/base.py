from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendInfo:
    name: str
    checkpoints: tuple
    sample_rate: int
    channels: int
    description: str
    supports_negative_prompt: bool
    pipeline_class: str
    training_objective: str
    license_note: str
    roles: tuple = ()
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "checkpoints": list(self.checkpoints),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "description": self.description,
            "supports_negative_prompt": self.supports_negative_prompt,
            "pipeline_class": self.pipeline_class,
            "training_objective": self.training_objective,
            "license_note": self.license_note,
            "roles": list(self.roles),
            "extra": dict(self.extra),
        }


def require_backend() -> None:
    try:
        import diffusers
        import transformers
    except ImportError as exc:
        from ..errors import DependencyError

        raise DependencyError(
            "Audio generation needs torch, diffusers, transformers, safetensors and "
            "soundfile. Run pip install audiyo and retry."
        ) from exc


class Backend:
    name = ""
    info: BackendInfo | None = None

    def load(self, checkpoint: str, **kwargs) -> Any:
        raise NotImplementedError

    def describe(self) -> dict:
        if self.info is None:
            from ..errors import ValidationError

            raise ValidationError("Backend " + repr(self.name) + " has no info.")
        return self.info.to_dict()
