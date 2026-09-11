# Security policy

## What Audiyo does with your data

Short version: it stays on your machine.

* Audio you generate or train on is read from disk and written back to disk. It is never uploaded. There is no telemetry, no hosted service, and no analytics.
* Your Hugging Face token is passed to the Hugging Face libraries for checkpoint access. It is never written to disk by Audiyo, never printed, and never included in logs. Error text from upstream is scrubbed for token patterns.
* Training checkpoints are loaded back with `weights_only=True`, so a tampered checkpoint file cannot run code on load.

## Reporting a vulnerability

Open an issue with SECURITY in the title, or contact the maintainers directly if public disclosure would put users at risk. Include:

1. What you did, step by step.
2. What you expected to happen.
3. What happened instead, with logs (remove your token first).
4. Your platform, Python version, and `pip freeze` output for the relevant packages.

We will confirm within a week, fix what we can, and credit you unless you prefer to stay anonymous.

## Dependency audit

Audited with `pip-audit` for the 0.0.1 release:

* torch, diffusers, transformers, safetensors, soundfile, numpy, peft, librosa, psutil: no known issues reported.
* accelerate: one advisory about path traversal in `load_checkpoint_in_model` and `load_checkpoint_and_dispatch`. Audiyo never calls those. Loading goes through the Diffusers pipeline with safetensors weights only. Keep accelerate updated.
* pillow, requests, urllib3 (indirect dependencies): advisories exist in older versions. Updating them is safe: `pip install -U pillow requests urllib3`.

Re-run the audit any time with:

```
pip install pip-audit
pip-audit --local
```

## Safe use checklist

* Keep Python and drivers current.
* Do not train on dataset folders you do not trust. Filenames and captions can end up in errors and logs.
* Do not share your HF token. Prefer `huggingface-cli login` over pasting tokens into scripts.
* Fetch `stabilityai/stable-audio-open-1.0` only from its official page. Audiyo refuses any other checkpoint id.
* Review adapters before loading them. Treat an adapter from an untrusted source like any unknown file from a stranger.
