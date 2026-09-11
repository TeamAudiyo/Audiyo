from __future__ import annotations

import os
import sys
import time


def describe(wav, sr):
    import numpy as np

    mono = wav.mean(axis=0) if wav.ndim == 2 else wav
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono))) > 0))
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-9))
    flat = float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9))
    return {"peak": round(peak, 4), "rms": round(rms, 4), "zcr": round(zcr, 4), "centroid_hz": round(centroid, 1), "flatness": round(flat, 4)}


def main():
    import soundfile as sf
    import torch
    from diffusers import AudioLDMPipeline

    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    outdir = "listen"
    os.makedirs(outdir, exist_ok=True)
    print("loading pipeline")
    t0 = time.perf_counter()
    pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float32)
    pipe.to("cpu")
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    print("loaded in " + str(round(time.perf_counter() - t0, 1)) + " s")
    clips = [
        ("rain_window", "rain falling against a window with distant thunder"),
        ("rain_forest", "steady rain in a forest, water dripping from leaves"),
    ]
    for name, prompt in clips:
        gen = torch.Generator(device="cpu").manual_seed(42)
        t1 = time.perf_counter()
        out = pipe(prompt=prompt, audio_length_in_s=duration, num_inference_steps=steps, generator=gen, output_type="np")
        secs = time.perf_counter() - t1
        wav = out.audios[0]
        path = os.path.join(outdir, name + ".wav")
        if wav.ndim == 1:
            data = wav
        elif wav.ndim == 2 and min(wav.shape) <= 2:
            data = wav.T if wav.shape[0] <= 2 else wav
        else:
            data = wav
        sf.write(path, data, 16000)
        info = describe(wav, 16000)
        print(name + " wrote " + path + " shape " + str(wav.shape) + " in " + str(round(secs, 1)) + " s")
        print("metrics " + str(info))


if __name__ == "__main__":
    main()
