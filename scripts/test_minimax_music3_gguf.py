from __future__ import annotations

import argparse
import json
import os
import sys
import time


DEFAULT_LYRICS = "[verse]\nMorning light on the water\n[chorus]\nWe run with the night"


def parse_args():
    p = argparse.ArgumentParser(description="Smoke test for Minimax-Music3 GGUF on Audiyo.")
    p.add_argument("--prompt", default="upbeat synth-pop with bright analog arps")
    p.add_argument("--lyrics", default=None)
    p.add_argument("--lyrics-file", default=None)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--memory-mode", default="balanced", choices=["performance", "balanced", "low", "minimal"])
    p.add_argument("--quant", default=None)
    p.add_argument("--checkpoint", default="TeamAudiyo/Minimax-Music3-GGUF")
    p.add_argument("--output", default="mm3_test.wav")
    p.add_argument("--compare-seed", type=int, default=None)
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def read_lyrics(args):
    if args.lyrics_file:
        with open(args.lyrics_file, "r", encoding="utf-8") as handle:
            return handle.read()
    if args.lyrics:
        return args.lyrics
    return DEFAULT_LYRICS


def main():
    args = parse_args()
    import torch

    from audiyo.estimate import estimate_requirements
    from audiyo.hardware import detect_hardware

    hw = detect_hardware("auto")
    print("torch: " + torch.__version__)
    print("cuda available: " + str(hw.cuda_available))
    print("gpu: " + str(hw.gpu_name) + " vram: " + str(hw.gpu_vram_gb))
    try:
        import diffusers

        print("diffusers: " + diffusers.__version__)
    except Exception as exc:
        print("diffusers import failed: " + str(exc)[:200])
        raise SystemExit(1)
    if not hw.cuda_available and not args.allow_cpu:
        print("No CUDA GPU found. Music generation on CPU is very slow.")
        print("Rerun with --allow-cpu if you really want to try, or move to a CUDA machine.")
        raise SystemExit(2)
    detail = estimate_requirements(args.checkpoint, args.memory_mode)
    print(json.dumps(detail, indent=2))
    lyrics = read_lyrics(args)
    print("lyrics lines: " + str(len(lyrics.splitlines())))
    from audiyo import AudioModel

    t0 = time.perf_counter()
    model = AudioModel.from_pretrained(
        args.checkpoint,
        device="auto",
        memory_mode=args.memory_mode,
        quant=args.quant,
    )
    print("loaded in " + str(round(time.perf_counter() - t0, 1)) + " s")
    print(json.dumps(model.describe_memory(), indent=2, default=str))
    pipe = model.pipeline
    print("fallback components: " + str(getattr(pipe, "_audiyo_fallback_components", [])))
    print("gguf file: " + str(getattr(pipe, "_audiyo_gguf_file", None)))
    print("gguf mapped tensors: " + str(getattr(pipe, "_audiyo_gguf_mapped", None)))
    print("stage mode: " + str(getattr(pipe, "_audiyo_stage_mode", None)))
    result = model.generate(
        prompt=args.prompt,
        lyrics=lyrics,
        duration_seconds=args.duration,
        seed=args.seed,
        num_inference_steps=args.steps,
    )
    result.save(args.output)
    print("wrote " + args.output)
    print("peak: " + str(round(result.peak, 4)))
    print("latency s: " + str(result.performance.get("latency_s")))
    print("call args: " + str(result.settings.get("call_args")))
    from audiyo.quality import analyze_file, compare_files, write_listening_sheet

    metrics = analyze_file(args.output)
    print(json.dumps(metrics, indent=2))
    if metrics["peak"] < 0.02:
        print("Quiet file. Try normalize on playback, it may still be fine.")
    if metrics["clipped_fraction"] > 0.01:
        print("Clipping detected. Try fewer steps or a gentler prompt.")
    entries = [{"file": os.path.abspath(args.output), "prompt": args.prompt, "seed": args.seed, "rating": "", "notes": ""}]
    if args.compare_seed is not None:
        second = os.path.splitext(args.output)[0] + "_seed" + str(args.compare_seed) + os.path.splitext(args.output)[1]
        alt = model.generate(
            prompt=args.prompt,
            lyrics=lyrics,
            duration_seconds=args.duration,
            seed=args.compare_seed,
            num_inference_steps=args.steps,
        )
        alt.save(second)
        print("wrote " + second)
        report = compare_files(args.output, second)
        print(json.dumps(report["deltas"], indent=2))
        entries.append({"file": os.path.abspath(second), "prompt": args.prompt, "seed": args.compare_seed, "rating": "", "notes": ""})
    sheet = os.path.splitext(args.output)[0] + "_sheet.csv"
    write_listening_sheet(entries, sheet)
    print("wrote " + sheet + ". Listen and fill in rating plus notes.")
    print("OK")


if __name__ == "__main__":
    main()
