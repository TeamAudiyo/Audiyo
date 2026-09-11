# Benchmarking

How to compare setups without fooling yourself.

## What to compare

Same checkpoint, prompts, duration, steps, and hardware:

1. Plain Diffusers pipeline with defaults.
2. Plain Diffusers pipeline with its documented optimizations on.
3. Audiyo in the memory mode you care about.

Keep batch size at 1 unless batching is what you are testing.

## Metrics

Audiyo reports per run:

* Latency in seconds, mean plus min and max.
* Audio seconds per wall second.
* Memory used before and after, plus peak use.
* Settings, dtype, device, and library versions.

The first run after load is always slower. Warm up at least once and report steady state numbers separately.

## Running it

```
python benchmarks/run.py balanced 10
```

This loads Audiyo in the given mode, warms up, averages repeats, and writes JSON. For a fair Diffusers baseline, load the pipeline directly in a separate process and run the same prompts. Keep the header when you share numbers or they cannot be compared.

## Quality checks

Dtype and tiling change output slightly. For anything you publish, listen to full vs optimized renders at matched seeds and say what you heard. Numbers alone are not enough.

## What is not claimed

This repo has no preset benchmark numbers. Run the script on your own hardware and share results with the header attached.
