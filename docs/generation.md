# Generation controls

Small options that save re-renders.

## Variations

Same prompt, several seeds:

```python
clips = model.generate_batch(["rain on a roof", "rain in a forest"], seed=42)
takes = model.generate_variations("rain on a roof", seeds=[1, 2, 3])
```

Or from the command line:

```
audiyo generate "rain on a roof" --seeds 1,2,3 -o rain.wav
```

That writes rain_seed1.wav, rain_seed2.wav, and so on.

## Clean endings

Fades, peak normalize, a soft limiter, and edge silence trim:

```python
result = model.generate(
    prompt="rain on a roof",
    duration_seconds=10,
    seed=42,
    fade_in_ms=20,
    fade_out_ms=200,
    normalize_peak=0.89,
    limiter=True,
    trim_silence=True,
)
```

Same flags exist on the CLI with the same names. Start with a short fade out. Only normalize when the raw file is too quiet. The limiter just keeps loud peaks under control.

## Formats

Save by extension:

```
result.save("rain.wav")
result.save("rain.flac")
result.save("rain.ogg")
```

Wav is the safe default. Flac is smaller with no loss. Ogg is smallest.

## Config files

Save settings once, reuse them:

```
audiyo init-config --out mysound.json
audiyo generate --config mysound.json
```

Command line flags still win over the file.
