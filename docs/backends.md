# Backends

One public API, one backend per model family. `AudioModel.from_pretrained`
keeps its shape. A dispatcher picks the backend from the checkpoint id,
and each backend lives in its own file under `src/audiyo/backends/`.

Current registry:

* `stable-audio` — Stable Audio Open, the only runnable backend.
* `testkit` — the tiny diffusion stand-in, for offline tests.
* `minimax-music` — Minimax-Music3, loads through
  `MiniMaxMusic3ModularPipeline` with GGUF DiT quants from
  `TeamAudiyo/Minimax-Music3-GGUF` plus
  sequential stage offloading. Default `q4_k_m` peaks under 4.5 GB.

Adding a backend means a new file with a `Backend` subclass, one
registry line, and capability info (sample rate, channels, negative
prompt support, training objective, license note). Nothing else in the
library changes.

## Evaluated: Minimax-Music3

Checked against `MiniMaxAI/MiniMax-Music3` and the diffusers docs.
Ungated, custom community license (UI attribution, authorization above
20M dollars yearly revenue, acceptable use policy). Not Apache or MIT,
but no worse in kind than the Stability terms Audiyo already accepts.

Architecture, all verified: an 8B Qwen3-based global language model
plus a 0.6B local depth decoder predict RVQ codebooks frame by frame
from lyrics and a music description. Fused hidden states feed a 2.4B
flow-matching transformer, and a DAC-style vocoder decodes stereo
audio. Lyrics use section tags like `[verse]` that must sit alone on
their line. Songs run up to about six minutes.

Two things to know before running it. Diffusers 0.40.0 or newer is
required (it ships `MiniMaxMusic3ModularPipeline`, which the old
pinned 0.39.0 lacked), and CUDA is needed for real generation.
The DiT can come from `TeamAudiyo/Minimax-Music3-GGUF` quantized GGUF weights
(default `q4_k_m`, 1.49 GB) for a peak under 4.5 GB, or from the
bf16 checkpoint (peak near 20.6 GB sequential, 10.3 GB with the
LLM in 8-bit). `load` applies
`enable_sequential_cpu_offload()` so the 8B LLM, DiT, and
vocoder swap in and out of GPU RAM sequentially, with the Audiyo
stage offloader as fallback. `src/audiyo/testkit/music.py` still
proves the interface offline: tag
parsing with the real drop rule plus deterministic stereo output.

One honest discrepancy: the repo advertises 32 kHz output while the
diffusers vocoder renders native 44.1 kHz and leaves resampling to the
caller. The proxy mirrors the 44.1 kHz side.

GGUF splits sometimes ship without the tokenizer, language model, or
depth decoder. When any of those come back empty, the loader pulls just
that piece from the matching subfolder in `MiniMaxAI/MiniMax-Music3`
and says which pieces it filled in. If a piece cannot load, it fails
with the piece named instead of generating wrong audio.

Each backend only gets the arguments it understands. Stable Audio keeps
its timing window, guidance, and negative prompt. Music gets prompt,
lyrics, length, steps, and seed, and runs its built-in guidance. Lyrics
are required for music, clips always start at zero, and one call makes
one clip. Songs run up to six minutes.

Revisit on a GPU machine with newer diffusers. LoRA targets and the
training objective stay marked unverified until an official fine-tune
script exists.

## Dual-component LoRA (Minimax-Music3 plan)

Music has two halves, so the adapter API takes a `target`: `llm` for
the 8B composer, `transformer` for the 2.4B voice, `both` for the full
job. Loading mirrors it, so a structure adapter and a timbre adapter
can combine at generation time.

The target module lists are assumed, not official: Qwen-style names
for the LLM side, diffusers DiT names for the flow side. The code
treats them that way. Every attach verifies the names against the
loaded modules first and refuses to inject into names that do not
exist, and saving records exactly what matched. `src/audiyo/testkit/`
carries Qwen-shaped and DiT-shaped proxy blocks so the freeze matrix
is tested for real: each target leaves the other side frozen, both
trains both, and save plus reload round-trips per side plus a manifest.

## Stage offloading (Minimax-Music3 plan)

Minimax-Music3 runs in stages, so Audiyo never needs the whole stack resident.
`src/audiyo/backends/music_stages.py` holds the plan plus a stage
offloader that keeps one stage on the device at a time: structure
(8B LLM plus 0.6B decoder), flow (2.4B transformer), vocoder. Estimates
assume 20 percent headroom over raw weight bytes:

* performance, everything resident in bfloat16: about 23 GB.
* balanced, sequential stages in bfloat16: about 20.6 GB peak.
* low, sequential with the LLM in 8-bit: about 10.3 GB peak.
* minimal, sequential with the LLM in 4-bit: about 6 GB peak.

Quantization is opt-in and needs bitsandbytes. It applies to the
language model on its own after the pipeline loads, never as a
pipeline-wide flag. Smaller cards also pick it automatically in
balanced mode. Which pieces got quantized is recorded on the loaded
pipeline.

One correction to an early sketch: 8 to 10 GB is the 8-bit figure, not
the 16-bit one. The 8B LLM alone is about 16 GB of weights in bfloat16,
so 16-bit sequential still peaks near 20 GB. Quantization is opt-in and
needs bitsandbytes. The flow stage already runs chunk by chunk, so
tiling buys little extra there. `TinyStagedMusicPipeline` proves the
scheduling pattern offline with real modules and tracked activation
order.

## Faster without quantizing

The slow part is the autoregressive stage, which normally emits one
token per forward pass. `src/audiyo/backends/music_decode.py` ships two
ways to emit several at once, both proven against plain greedy decoding
in `tests/test_music_decode.py`:

* Speculative decoding drafts a few tokens with a cheap model, checks
  them with one batched target pass, and keeps the accepted prefix plus
  a bonus token. Output is identical to greedy decoding with far fewer
  target calls. Needs a small draft model next to the 8B global LLM.
* Jacobi decoding re-predicts every position at once each sweep until
  nothing changes. Converged output equals greedy decoding. Wall-clock
  wins need a backend that truly batches.
* Chunk overlap planning lays the language model and flow stages over
  each other frame by frame instead of running them back to back. The
  planner and its span math are tested. Real gains need threaded
  execution on the real backend.

Still unevaluated, listed so nobody guesses: fewer flow Euler steps,
guidance caching, and compiling the stages. Those wait for a GPU
machine with the real weights.
