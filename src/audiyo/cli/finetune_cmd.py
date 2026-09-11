from __future__ import annotations


def run_finetune(args) -> int:
    from ..model import AudioModel

    model = AudioModel.from_pretrained(
        checkpoint=args.checkpoint,
        device=args.device,
        memory_mode=args.memory_mode,
        dtype=args.dtype,
        token=args.token,
    )
    report = model.finetune(
        dataset=args.dataset,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        duration_seconds=args.duration,
        seed=args.seed,
    )
    print("Done. Final loss " + str(round(report.final_loss, 4)))
    print("Steps " + str(report.steps) + " in " + str(round(report.elapsed_s, 1)) + " s")
    print("Checks " + str(report.checks))
    print("Adapter saved to " + args.output_dir + "/adapter")
    return 0
