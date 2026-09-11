# Troubleshooting

## Access denied or 401/403 errors

The license gate is not accepted or you are not logged in. Open https://huggingface.co/stabilityai/stable-audio-open-1.0 while logged in, accept the terms, then run `huggingface-cli login`. If you use `HF_TOKEN`, make sure it belongs to the same account. Do not paste tokens into issues.

## Unsupported checkpoint

Audiyo 0.0.1 loads one checkpoint id only. Anything else raises a clear error. This is on purpose.

## Out of memory

First shorten duration and lower steps to check the setup works. Then move down the presets: performance to balanced to low to minimal. Keep batch size at 1. Close other heavy apps. Lower presets trade speed for fit. If memory is the limit, shorter audio is the fix.

## float16 on CPU

Rejected up front. It is slow and poorly supported here. Use float32 on CPU or move to a supported setup.

## No supported processor found

`torch.cuda.is_available()` returned False. Common causes: a CPU-only install, missing drivers, or a container without access to the hardware. Check the install and drivers first.

## Audio errors

File not found names the path. Bad extensions list what is accepted. Decode failures name the file. Missing captions name the expected filename, since it is often a typo.

Resampling needs librosa. Install with:

```
pip install audiyo[train]
```

## Bad loss in training

Often a broken clip or a learning rate set too high. Try `limit=2` on known good files first. The training report shows whether grads flowed and params moved.

## Adapter refuses to load

`load_adapter` checks the base model id in `audiyo_adapter.json` and refuses mismatches. Train and serve on the same checkpoint.

## Slow CPU generation

Expected. The CPU path is fine for tests and small checks. Practical generation needs stronger hardware.
