from __future__ import annotations

import json


def run_estimate(args) -> int:
    from ..estimate import estimate_requirements

    detail = estimate_requirements(
        checkpoint=getattr(args, "checkpoint", "stabilityai/stable-audio-open-1.0"),
        memory_mode=getattr(args, "memory_mode", "balanced"),
        gpu_vram_gb=getattr(args, "vram", None),
    )
    print(json.dumps(detail, indent=2))
    verdict = detail.get("verdict", "")
    if verdict == "fits":
        print("Looks fine on this card. Real use still moves with length and steps.")
    elif verdict == "tight":
        print("Close to the limit. Use a shorter clip or a lighter mode first.")
    elif verdict == "unlikely":
        print("Likely too big as asked. Pick low or minimal, or a smaller quant.")
    return 0


def run_ui(args) -> int:
    from ..ui import launch_ui

    launch_ui(
        port=int(getattr(args, "port", 7860)),
        share=bool(getattr(args, "share", False)),
        checkpoint=str(getattr(args, "checkpoint", "stabilityai/stable-audio-open-1.0")),
    )
    return 0


def run_init_config(args) -> int:
    from ..configfile import write_example_config

    path = write_example_config(getattr(args, "out", "audiyo.json"))
    print("Wrote " + str(path) + ". Edit it, then run audiyo generate --config " + str(path))
    return 0
