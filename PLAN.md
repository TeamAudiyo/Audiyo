# Audiyo implementation plan

Notes on decisions made before writing code, so future changes can be checked against what was verified.

## Backend

Checkpoint: `stabilityai/stable-audio-open-1.0`. It is the only Stable Audio release with open weights, a Diffusers pipeline, and a published architecture description.

Checked against Diffusers docs, the Hugging Face model card, the paper (arXiv 2407.14358), and the `stable-audio-tools` repo:

* Pipeline class is `diffusers.StableAudioPipeline`. Parts are `vae`, `text_encoder` (T5), `projection_model`, `tokenizer`, `transformer`, and `scheduler`.
* Audio is stereo, 44100 Hz, up to about 47 seconds. The exact limit is read from the loaded pipeline, not hardcoded.
* `__call__` takes `prompt`, `negative_prompt`, timing bounds, steps, guidance scale, and generator. Negative prompts work through classifier-free guidance.
* Access is gated. Users accept the Stability AI Community License on the model page and log in with a Hugging Face token. Audiyo never bundles weights.
* License: weights under Stability AI Community License. Audiyo code is Apache-2.0. LoRA weights derived from the model inherit the Community License terms.

What Diffusers already provides, so Audiyo does not rebrand it: CPU offload, sequential offload, attention slicing, xFormers hooks, SDPA, VAE slicing and tiling, dtype selection, device_map, and quantization examples. Audiyo presets combine these calls and report what ran.

## LoRA targets and training objective

No first-party LoRA support exists for this pipeline in the checked Diffusers version. The DiT attention linears are named `to_q`, `to_k`, `to_v`, and `to_out`, so Audiyo defaults to `["to_q", "to_k", "to_v", "to_out.0"]` but checks the names exist before injecting anything.

Training uses v-prediction, as in the paper and the training tools repo:

```
alpha, sigma = cos(t*pi/2), sin(t*pi/2)
noised = x * alpha + eps * sigma
target = eps * alpha - x * sigma
loss = MSE(model(noised, t, cond), target)
```

Timesteps are continuous in [0,1]. The paper does not name the sampler, so Audiyo uses uniform and documents it. Text comes from frozen t5-base. Timing conditioning must reflect the true clip length with silence padding to the full window.

## Memory presets

Four presets: performance, balanced, low, minimal. They differ only in dtype, offload, VAE slicing and tiling, and attention slicing. Quantization stays opt-in and experimental. Offload moves idle parts aside, and Audiyo reports what each preset did.

Chunked VAE decoding is experimental. Audiyo adds a helper that compares full vs tiled decode on the same latents.

## Scope cuts

* One model only. Anything else raises a clear error.
* No bundled weights, no telemetry, no hosted services.
* Benchmarks ship as runnable scripts with no preset numbers.
* Tests run on CPU. Practical CPU generation speed is not claimed.
