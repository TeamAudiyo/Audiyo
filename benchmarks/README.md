# Benchmarks

Runnable comparisons live here, not pasted numbers.

Run one mode:

```
python benchmarks/run.py balanced 10
```

This writes `benchmark_balanced.json` with checkpoint, device, dtype, library versions, plus per prompt latency and memory.

Compare modes by running each in its own process. Keep duration, steps, prompts, and seeds the same.
