from __future__ import annotations

import os
import sys

import numpy as np


def main():
    import soundfile as sf

    out = sys.argv[1] if len(sys.argv) > 1 else "demo_data"
    os.makedirs(out, exist_ok=True)
    sr = 44100
    for name, freq in [("rain", 440.0), ("cafe", 220.0)]:
        length = sr * 4
        tone = (0.2 * np.sin(2 * np.pi * freq * np.arange(length) / sr)).astype(np.float32)
        stereo = np.stack([tone, tone], axis=1)
        sf.write(os.path.join(out, name + ".wav"), stereo, sr)
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(name + " test tone for smoke tests")
    print("Wrote demo dataset to " + out)


if __name__ == "__main__":
    main()
