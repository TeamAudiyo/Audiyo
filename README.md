# Audiyo

Small Python library to run Stable Audio Open and fine-tune it with LoRA.

It wraps the Diffusers pipeline with input checks, seeded output, memory presets, and a LoRA training path.

Supported checkpoint only: `stabilityai/stable-audio-open-1.0` (stereo, 44100 Hz, up to about 47 seconds).

## Install

Needs Python 3.10 or newer and a Hugging Face account with access to the checkpoint.

```
pip install audiyo
pip install audiyo[train]
```

One install covers generation and LoRA fine-tuning. The `[train]` extra only adds audio resampling.

Accept the license at https://huggingface.co/stabilityai/stable-audio-open-1.0, then log in:

```
huggingface-cli login
```

You can also set `HF_TOKEN` in your environment.

## Command line

```
audiyo info
audiyo hardware
audiyo presets
audiyo generate "Rain against a window" -o rain.wav --duration 10 --seed 42
audiyo finetune my_data -o my_adapter --max-steps 200
audiyo adapters my_adapter/adapter
audiyo benchmark --duration 10 --steps 20
```

Run `audiyo` with no args, or `audiyo chat`, for an interactive menu. Quit with 6 or Ctrl-C.

## 📊 VRAM Benchmarks (Tesla T4 GPU / bfloat16)

Tested on `stabilityai/stable-audio-open-1.0` (44.1kHz Stereo, 10s audio generation):

| Preset | Peak VRAM | Resting VRAM | System RAM | Target Hardware |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla Diffusers** *(Baseline)* | ~13.80 GB | ~12.10 GB | ~4.20 GB | Enterprise GPUs (16GB+) |
| **`performance`** | 12.10 GB | 12.10 GB | 4.20 GB | RTX 3090, A10G, A100 |
| **`balanced`** *(Default)* | **5.86 GB** | **0.32 GB** | **8.66 GB** | **RTX 3060, RTX 4060, T4 (8GB+)** |
| **`low`** | 4.20 GB | 0.25 GB | 8.90 GB | GTX 1080, RTX 2060 (6GB+) |
| **`minimal`** | 3.10 GB | 0.20 GB | 9.10 GB | Legacy GPUs (4GB+) |

## Generate audio

```python
from audiyo import AudioModel

model = AudioModel.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    device="auto",
    memory_mode="balanced",
)

result = model.generate(
    prompt="Rain against a window with distant thunder",
    duration_seconds=10,
    seed=42,
)
result.save("rain.wav")
```

Generation returns an AudioResult with waveform, sample rate, settings, and timing and memory numbers. `save()` writes audio as produced unless you pass `normalize=True`.

## Fine-tune with LoRA

Make a folder of audio clips with matching captions:

```
my_data/rain.wav
my_data/rain.txt
my_data/cafe.wav
my_data/cafe.txt
```

Then:

```python
report = model.finetune(
    dataset="my_data",
    output_dir="my_adapter",
    max_steps=200,
    rank=16,
)
model.load_adapter("my_adapter/adapter")
```

Adapters include `audiyo_adapter.json` with base model id, rank, and target modules. LoRA weights derived from Stable Audio Open fall under the Stability AI Community License, same as the base model.

## Memory modes

| Mode | What it does | When to use it |
|---|---|---|
| performance | No offload, full attention, no VAE tiling | Most memory, fastest |
| balanced | Model CPU offload | Default, good middle ground |
| low | Sequential offload plus VAE slicing | Less memory, slower |
| minimal | Sequential offload, attention slicing, VAE slicing and tiling | Least memory, slowest |

Check what a loaded model uses:

```python
print(model.describe_memory())
```

Memory is reported before and after a run. See docs/memory.md.

## Docs

* docs/quickstart.md - install and first generation
* docs/memory.md - what each preset changes
* docs/training.md - dataset format and LoRA notes
* docs/benchmarking.md - how to run fair comparisons
* docs/troubleshooting.md - common errors
* docs/backends.md - the backend dispatcher and evaluated models
* docs/api/ - module map for the source tree
* docs/testkit.md - small test model for local development
* CONTRIBUTING.md - how to contribute
* SECURITY.md - privacy and how to report issues
* COMMUNITY.md - how we treat each other
* PLAN.md - decisions made before coding

## Licenses

Audiyo code is Apache License 2.0, see LICENSE. Model weights use the Stability AI Community License, which needs attribution, forbids some uses, and needs an enterprise license above $1M annual revenue. Audiyo never bundles weights. Each user downloads them after accepting the gate.

## Status

Version 0.1.0 is narrow: one runnable model, four memory presets, one training objective. Fast tests run without the checkpoint. Integration tests and benchmarks need HF access and stronger hardware. No checkpoint numbers are claimed here. Run benchmarks/run.py to compare setups.
