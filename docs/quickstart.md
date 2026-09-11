# Getting started

## 1. Install

Needs Python 3.10 or newer.

```
pip install audiyo
```

For training:

```
pip install audiyo[train]
```

For benchmarks:

```
pip install audiyo[bench]
```

To work from this repo:

```
pip install -e ".[all]"
```

Importing audiyo is cheap. It loads no models and touches no network. That happens inside `AudioModel.from_pretrained`.

## 2. Get access

Stable Audio Open is gated. Open this page while logged in:

https://huggingface.co/stabilityai/stable-audio-open-1.0

Accept the license, then log in locally:

```
huggingface-cli login
```

You can also set `HF_TOKEN`. Short license version: research and non-commercial use is free, commercial use needs registration, and revenue above $1M needs an enterprise license from Stability.

## 3. First generation

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

Notes:

* Duration is checked before generation. The limit comes from the loaded pipeline, about 47 seconds.
* Seed gives repeat output on the same machine and versions. It does not promise identical output across hardware or library versions.
* Negative prompts work through guidance. Default guidance is 7.
* Result holds waveform, sample rate, settings, and timing plus memory numbers.

## 4. Picking a memory mode

Start with balanced. If you run out of memory, go to low, then minimal. If you have memory to spare, try performance.

```python
print(model.describe_memory())
```

Offload moves idle parts aside. Total use stays about the same.

## 5. Next steps

To train, read docs/training.md. To compare fairly, read docs/benchmarking.md. When something breaks, read docs/troubleshooting.md.
