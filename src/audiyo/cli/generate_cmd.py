from __future__ import annotations

import json


def run_generate(args) -> int:
    from ..model import AudioModel

    prompt = args.prompt
    if not prompt:
        prompt = input("Prompt: ").strip()
        if not prompt:
            print("No prompt given, nothing to do.")
            return 2
    model = AudioModel.from_pretrained(
        checkpoint=args.checkpoint,
        device=args.device,
        memory_mode=args.memory_mode,
        dtype=args.dtype,
        token=args.token,
    )
    result = model.generate(
        prompt=prompt,
        duration_seconds=args.duration,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        negative_prompt=args.negative_prompt,
    )
    result.save(args.output)
    print("Wrote " + args.output)
    print("Peak amplitude " + str(round(result.peak, 4)))
    print("Latency " + str(result.performance.get("latency_s")) + " s")
    print(json.dumps(result.performance.get("cuda_after"), indent=2, default=str))
    return 0
