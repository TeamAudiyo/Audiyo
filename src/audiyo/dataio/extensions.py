from __future__ import annotations

import os

SUPPORTED_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus")
AUDIO_EXTS = SUPPORTED_EXTENSIONS


def extension_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def is_supported_audio(path: str) -> bool:
    return extension_of(path) in SUPPORTED_EXTENSIONS
