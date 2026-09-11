from __future__ import annotations

import numpy as np


def test_cache_key_changes():
    from audiyo.datasets import cache_key

    a = cache_key("model-a", {"sr": 44100})
    b = cache_key("model-b", {"sr": 44100})
    assert a != b


def test_cache_key_stable():
    from audiyo.datasets import cache_key

    a = cache_key("model-a", {"sr": 44100})
    b = cache_key("model-a", {"sr": 44100})
    assert a == b


def test_dataio_windows_match_audio():
    import numpy as np

    from audiyo.audio import crop_or_pad as old_crop
    from audiyo.dataio import crop_or_pad as new_crop

    wav = np.ones((2, 100), dtype=np.float32)
    a, _ = old_crop(wav, 200)
    b, _ = new_crop(wav, 200)
    assert a.shape == b.shape


def test_extensions_helper():
    from audiyo.dataio import is_supported_audio

    assert is_supported_audio("clip.wav") is True
    assert is_supported_audio("notes.txt") is False
