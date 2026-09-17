from __future__ import annotations

import json


def run_generate(args) -> int:
    import json
    import os

    from ..configfile import filter_generate_config, load_config_file, resolve_options
    from ..model import AudioModel

    defaults = {
        "prompt": None,
        "output": "output.wav",
        "duration": 10.0,
        "seed": 42,
        "steps": 100,
        "guidance": 7.0,
        "negative_prompt": None,
        "lyrics": None,
        "fade_in_ms": 0.0,
        "fade_out_ms": 0.0,
        "normalize_peak": None,
        "limiter": False,
        "trim_silence": False,
        "checkpoint": "stabilityai/stable-audio-open-1.0",
        "device": "auto",
        "memory_mode": "balanced",
        "dtype": None,
        "token": None,
    }
    cli_values = {
        "prompt": getattr(args, "prompt", None),
        "output": getattr(args, "output", "output.wav"),
        "duration": getattr(args, "duration", 10.0),
        "seed": getattr(args, "seed", 42),
        "steps": getattr(args, "steps", 100),
        "guidance": getattr(args, "guidance", 7.0),
        "negative_prompt": getattr(args, "negative_prompt", None),
        "lyrics": getattr(args, "lyrics", None),
        "fade_in_ms": getattr(args, "fade_in_ms", 0.0),
        "fade_out_ms": getattr(args, "fade_out_ms", 0.0),
        "normalize_peak": getattr(args, "normalize_peak", None),
        "limiter": getattr(args, "limiter", False),
        "trim_silence": getattr(args, "trim_silence", False),
        "checkpoint": getattr(args, "checkpoint", defaults["checkpoint"]),
        "device": getattr(args, "device", "auto"),
        "memory_mode": getattr(args, "memory_mode", "balanced"),
        "dtype": getattr(args, "dtype", None),
        "token": getattr(args, "token", None),
    }
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = filter_generate_config(load_config_file(config_path))
        merged = resolve_options(defaults, cli_values, cfg)
    else:
        merged = cli_values
    prompt = merged["prompt"]
    if not prompt:
        prompt = input("Prompt: ").strip()
        if not prompt:
            print("No prompt given, nothing to do.")
            return 2
    model = AudioModel.from_pretrained(
        checkpoint=merged["checkpoint"],
        device=merged["device"],
        memory_mode=merged["memory_mode"],
        dtype=merged["dtype"],
        token=merged["token"],
    )
    seeds_raw = getattr(args, "seeds", None)
    if seeds_raw:
        try:
            seeds = [int(x.strip()) for x in str(seeds_raw).split(",") if x.strip()]
        except Exception:
            print("Could not parse seeds, use like 1,2,3.")
            return 2
        results = model.generate_variations(
            prompt=prompt,
            seeds=seeds,
            duration_seconds=merged["duration"],
            num_inference_steps=merged["steps"],
            guidance_scale=merged["guidance"],
            negative_prompt=merged["negative_prompt"],
            lyrics=merged["lyrics"],
            fade_in_ms=merged["fade_in_ms"],
            fade_out_ms=merged["fade_out_ms"],
            normalize_peak=merged["normalize_peak"],
            limiter=merged["limiter"],
            trim_silence=merged["trim_silence"],
        )
        base, ext = os.path.splitext(merged["output"])
        if not ext:
            ext = ".wav"
        for result, seed in zip(results, seeds):
            path = base + "_seed" + str(seed) + ext
            result.save(path)
            print("Wrote " + path + " peak " + str(round(result.peak, 4)))
        return 0
    result = model.generate(
        prompt=prompt,
        duration_seconds=merged["duration"],
        seed=merged["seed"],
        num_inference_steps=merged["steps"],
        guidance_scale=merged["guidance"],
        negative_prompt=merged["negative_prompt"],
        lyrics=merged["lyrics"],
        fade_in_ms=merged["fade_in_ms"],
        fade_out_ms=merged["fade_out_ms"],
        normalize_peak=merged["normalize_peak"],
        limiter=merged["limiter"],
        trim_silence=merged["trim_silence"],
    )
    result.save(merged["output"])
    print("Wrote " + str(merged["output"]))
    print("Peak amplitude " + str(round(result.peak, 4)))
    print("Latency " + str(result.performance.get("latency_s")) + " s")
    print(json.dumps(result.performance.get("cuda_after"), indent=2, default=str))
    return 0
