from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ValidationError

SUPPORTED_CHECKPOINT = "stabilityai/stable-audio-open-1.0"
MUSIC3_CHECKPOINT = "MiniMaxAI/MiniMax-Music3"
MM3_GGUF_REPO = "TeamAudiyo/MM3-GGUF"
MM3_GGUF_FILES = (
    "MiniMax-Music3-Q3_K_M.gguf",
    "MiniMax-Music3-Q4_K_M.gguf",
    "MiniMax-Music3-Q5_K_M.gguf",
    "MiniMax-Music3-Q6_K.gguf",
    "MiniMax-Music3-Q8_0.gguf",
    "MiniMax-Music3-F16.gguf",
)
SUPPORTED_CHECKPOINTS = (SUPPORTED_CHECKPOINT, MUSIC3_CHECKPOINT, MM3_GGUF_REPO) + tuple(
    MM3_GGUF_REPO + ":" + f for f in MM3_GGUF_FILES
)

SAMPLE_RATE = 44100
NUM_CHANNELS = 2
MAX_DURATION_SECONDS = 47.55
MIN_DURATION_SECONDS = 0.5
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_NUM_STEPS = 100
DEFAULT_GUIDANCE = 7.0

MemoryMode = Literal["performance", "balanced", "low", "minimal"]
MEMORY_MODES: tuple[str, ...] = ("performance", "balanced", "low", "minimal")


def check_checkpoint(checkpoint: str) -> str:
    if checkpoint not in SUPPORTED_CHECKPOINTS:
        raise ValidationError(
            f"Unsupported checkpoint {checkpoint!r}. Audiyo 0.1.0 supports only "
            f"{list(SUPPORTED_CHECKPOINTS)}. Loading anything else would need "
            "a different pipeline and conditioning setup, so it fails here "
            "instead of producing wrong audio."
        )
    return checkpoint


@dataclass
class GenerationConfig:
    prompt: str
    duration_seconds: float = DEFAULT_DURATION_SECONDS
    seed: int | None = None
    num_inference_steps: int = DEFAULT_NUM_STEPS
    guidance_scale: float = DEFAULT_GUIDANCE
    negative_prompt: str | None = None
    num_waveforms_per_prompt: int = 1
    audio_start_in_s: float = 0.0
    eta: float = 0.0
    fade_in_ms: float = 0.0
    fade_out_ms: float = 0.0
    normalize_peak: float | None = None
    limiter: bool = False
    trim_silence: bool = False
    trim_db: float = -50.0

    def validate(self, max_duration: float = MAX_DURATION_SECONDS) -> "GenerationConfig":
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValidationError("prompt must be a non-empty string.")
        if len(self.prompt) > 5000:
            raise ValidationError("prompt is too long; keep it under 5000 characters.")
        d = self.duration_seconds
        if not isinstance(d, (int, float)) or not (MIN_DURATION_SECONDS <= d <= max_duration):
            raise ValidationError(
                f"duration_seconds must be between {MIN_DURATION_SECONDS} and "
                f"{max_duration:.2f} for this checkpoint. Got {d!r}. Audiyo does "
                "not clamp this silently because the wrong length would change "
                "timing conditioning."
            )
        if self.audio_start_in_s < 0 or self.audio_start_in_s >= float(d):
            raise ValidationError("audio_start_in_s must be >= 0 and below duration_seconds.")
        if not isinstance(self.num_inference_steps, int) or not 1 <= self.num_inference_steps <= 500:
            raise ValidationError("num_inference_steps must be an int between 1 and 500.")
        if not 0.0 <= self.guidance_scale <= 30.0:
            raise ValidationError("guidance_scale must be between 0 and 30.")
        if self.negative_prompt is not None and not isinstance(self.negative_prompt, str):
            raise ValidationError("negative_prompt must be a string or None.")
        if self.num_waveforms_per_prompt not in (1, 2, 3, 4):
            raise ValidationError("num_waveforms_per_prompt must be 1, 2, 3, or 4.")
        if not 0.0 <= self.eta <= 1.0:
            raise ValidationError("eta must be between 0 and 1.")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValidationError("seed must be an int or None.")
        from .postfx import validate_postfx

        validate_postfx(
            self.fade_in_ms,
            self.fade_out_ms,
            self.normalize_peak,
            self.limiter,
            self.trim_silence,
            self.trim_db,
            self.duration_seconds,
        )
        return self


@dataclass
class FinetuneConfig:
    dataset: str
    output_dir: str
    method: str = "lora"
    max_steps: int = 1000
    rank: int = 16
    alpha: int = 16
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    mixed_precision: str | None = "fp16"
    seed: int = 0
    validation_split: float = 0.0
    cache_latents: bool = False
    cache_text_embeddings: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "FinetuneConfig":
        if self.method != "lora":
            raise ValidationError(
                f"Unsupported fine-tuning method {self.method!r}. Only 'lora' is "
                "implemented in this release."
            )
        if not self.dataset:
            raise ValidationError("dataset must be a path, CSV file, or HF dataset name.")
        if not self.output_dir:
            raise ValidationError("output_dir must be a non-empty path.")
        if self.max_steps < 1:
            raise ValidationError("max_steps must be at least 1.")
        if self.rank < 1 or self.rank > 128:
            raise ValidationError("rank must be between 1 and 128.")
        if self.alpha < 1:
            raise ValidationError("alpha must be positive.")
        if not 0.0 < self.learning_rate < 1.0:
            raise ValidationError("learning_rate looks wrong; expected a small float.")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValidationError("batch_size and gradient_accumulation_steps must be >= 1.")
        if self.mixed_precision not in (None, "no", "fp16", "bf16"):
            raise ValidationError("mixed_precision must be one of None, 'no', 'fp16', 'bf16'.")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValidationError("validation_split must be in [0, 1).")
        return self
