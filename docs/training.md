# Training a LoRA adapter

This path is experimental. It follows the published v-prediction objective. Good for plumbing checks and small runs.

## How the loss works

Each step encodes a clip to latents, samples t in [0, 1], noises the latents, and asks the transformer to predict the velocity target. Training uses the same text plus timing conditioning as inference.

Two assumptions are documented in code:

* Timesteps are uniform.
* CFG dropout is 0.1, so one step in ten trains without text.

Base weights stay frozen. Only LoRA params train.

## Dataset format

A folder with audio plus captions:

```
my_data/
  rain.wav
  rain.txt
  cafe.wav
  cafe.txt
```

Each txt file holds one caption. Supported types: wav, flac, ogg, mp3, m4a, opus. Clips are resampled, made stereo, then cropped or padded to the training window.

A CSV works too, with `audio_path` and `caption` columns. Paths are relative to the CSV file unless absolute.

Missing captions, empty files, and bad decodes raise errors that name the file.

## Caching

You can cache text embeddings and VAE latents to skip repeated work. The cache key includes the checkpoint id and preprocessing config.

Tradeoffs:

* Cached crops are fixed.
* Cached latents skip some sampling diversity.
* If you want max diversity, skip caching. If you want faster repeats, turn it on.

## Running

```python
report = model.finetune(
    dataset="my_data",
    output_dir="my_adapter",
    max_steps=200,
    rank=16,
    alpha=16,
    learning_rate=1e-4,
    duration_seconds=10,
)
```

Try `limit` on a few files first. The report shows loss, param counts, time, and checks. All checks should pass before you trust the adapter.

## Saving and reloading

Adapters live under `output_dir/adapter` with weights plus `audiyo_adapter.json`. That file records base model, rank, targets, and version. Loading refuses a mismatched base model.

```python
model.load_adapter("my_adapter/adapter")
```

Trainer state sits next to the adapter. Rerunning with the same output dir resumes from there.

## Targets

Defaults are `to_q`, `to_k`, `to_v`, and `to_out.0`. They are checked against the loaded model before use.
