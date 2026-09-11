# Memory

## Where memory goes

Stable Audio Open has three parts: text encoder, transformer, and stereo VAE. Peak use is often the transformer pass plus the VAE decode at the end.

Audiyo reports memory before and after a run, so you can see what generation actually used.

Offload moves idle parts aside. Total use stays about the same.

## Presets

All presets use calls Diffusers already provides. Audiyo adds checks, reporting, and defaults.

performance

* Offload: none.
* Attention slicing: off. VAE slicing and tiling: off.
* Fastest. Needs the most memory.

balanced (default)

* Offload: model CPU offload.
* Attention slicing: off. VAE slicing and tiling: off.
* Good starting point for most setups.

low

* Offload: sequential.
* VAE slicing: on.
* Slower. Fits tighter setups.

minimal

* Offload: sequential. Attention slicing: on. VAE slicing and tiling: on.
* Slowest. Tiled decode lowers peak use but can leave faint seams. Listen before keeping output made this way.

Dtype is separate from the preset. Lower precision halves weight memory roughly and changes numerics slightly.

## Overrides

```python
model = AudioModel.from_pretrained(device="auto", memory_mode="low")
print(model.describe_memory())
```

You can also pass explicit VAE tiling and attention slicing flags for tests. They override the preset and show up in the result.

## What is experimental

* VAE tiling. Audiyo exposes it in minimal mode and in `compare_full_vs_tiled`, which decodes the same latents both ways and reports the difference.
* Attention slicing on this pipeline. On only in minimal mode.
* Quantization. Off by default. If you try it, test quality yourself.

## Practical advice

Shorter audio and fewer steps save more memory than any preset. Keep batch size at 1 when memory is tight. Warm up once, then average a few runs.
