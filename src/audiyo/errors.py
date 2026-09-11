from __future__ import annotations


class AudiyoError(Exception):
    """Base error for everything Audiyo raises."""


class ValidationError(AudiyoError, ValueError):
    """Raised when user input fails checks before any heavy work starts."""


class CheckpointError(AudiyoError):
    """Raised when a checkpoint id is unsupported, missing, or unreadable."""


class AuthError(AudiyoError):
    """Raised when Hugging Face authentication or gated access fails."""


class DeviceError(AudiyoError):
    """Raised for unsupported or unavailable device/dtype combinations."""


class DependencyError(AudiyoError):
    """Raised when an optional dependency is needed but not installed."""


def _hint(message: str, hint: str) -> str:
    return f"{message}\nHint: {hint}"


def scrub_text(text: str) -> str:
    import os
    import re

    clean = str(text)
    token = os.environ.get("HF_TOKEN", "")
    if token and len(token) > 6:
        clean = clean.replace(token, "[redacted]")
    clean = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[redacted]", clean)
    clean = re.sub(r"(hf_[A-Za-z0-9]+)", "[redacted]", clean)
    return clean


def auth_hint(detail: str = "") -> AuthError:
    base = (
        "Could not access the checkpoint. Stable Audio Open is gated: "
        "accept the license at "
        "https://huggingface.co/stabilityai/stable-audio-open-1.0 and log in with "
        "`huggingface-cli login` or the HF_TOKEN environment variable."
    )
    if detail:
        base += f"\nUnderlying error: {scrub_text(detail)}"
    return AuthError(base)


def oom_hint(requested: str = "") -> AudiyoError:
    msg = "Out of memory during audio generation."
    if requested:
        msg += f" ({requested})"
    return AudiyoError(
        msg
        + "\nHint: try memory_mode='low' or 'minimal', a shorter "
        "duration_seconds, fewer inference steps, or num_waveforms_per_prompt=1. "
        "See docs/memory.md for what each preset changes."
    )
