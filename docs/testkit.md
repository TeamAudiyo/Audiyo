# Testkit: small test model

The real checkpoint needs gated access and stronger hardware. The testkit gives you a tiny model that uses the same code paths for local runs.

## What it is

`src/audiyo/testkit/` holds a small model with the same layout as Stable Audio Open: waveform VAE, small text encoder, 6 layer transformer, and v-prediction training.

What it is not: a substitute for real weights. Its audio sounds like nothing. It proves plumbing, not quality.

## What it checks

* `AudioModel.generate` end to end: checks, seeding, stereo output, metadata, WAV export.
* LoRA training end to end: frozen base, adapter grads, save, reload, resume.
* Same-seed determinism.
* Same-process reload without duplicate wrappers.
* Resume guard against overwriting a finished adapter.
* Tiled vs full decode on the same latents.
* Default LoRA targets against realistic layer names.

## Use it

```
python scripts/smoke_tiny.py
```

This prints param counts, writes a demo wav, compares tiled decoding, and runs a short LoRA train. Tests live in `tests/test_testkit.py` and need no downloads or tokens.
