from __future__ import annotations

import json


def run_info(args) -> int:
    from .._version import __version__
    from ..config import SUPPORTED_CHECKPOINT, SAMPLE_RATE, MAX_DURATION_SECONDS
    from ..memory import describe_presets

    print("audiyo " + __version__)
    print("checkpoint: " + SUPPORTED_CHECKPOINT)
    print("audio: stereo, " + str(SAMPLE_RATE) + " Hz, up to " + str(MAX_DURATION_SECONDS) + " s")
    print("presets: " + ", ".join(sorted(describe_presets())))
    return 0


def run_hardware(args) -> int:
    from ..hardware import detect_hardware, default_memory_mode

    hw = detect_hardware("auto")
    print(json.dumps(hw.__dict__, indent=2, default=str))
    mode, why = default_memory_mode(hw.device, hw.gpu_vram_gb)
    print("Suggested memory mode: " + mode)
    print(why)
    return 0


def run_presets(args) -> int:
    from ..memory import describe_presets

    for name, spec in describe_presets().items():
        print(name + ": " + spec["description"])
        print("  tradeoff: " + spec["tradeoff"])
    return 0


def run_adapters(args) -> int:
    from ..lora import read_adapter_meta

    meta = read_adapter_meta(args.adapter_dir)
    print(json.dumps(meta, indent=2, default=str))
    return 0


def run_bench(args) -> int:
    from ..benchmark import run_benchmark
    from ..model import AudioModel

    model = AudioModel.from_pretrained(
        checkpoint=args.checkpoint,
        device=args.device,
        memory_mode=args.memory_mode,
        dtype=args.dtype,
        token=args.token,
    )
    report = run_benchmark(
        model,
        ["Rain against a window with distant thunder"],
        duration_seconds=args.duration,
        num_inference_steps=args.steps,
        repeat=args.repeat,
    )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("Wrote " + args.out)
    else:
        print(text)
    return 0
