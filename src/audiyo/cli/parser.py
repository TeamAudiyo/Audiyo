from __future__ import annotations

import argparse

from ..config import SUPPORTED_CHECKPOINT


def common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", default=SUPPORTED_CHECKPOINT)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--memory-mode", default="balanced", choices=["performance", "balanced", "low", "minimal"])
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--token", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiyo", description="Memory-efficient text-to-audio with Stable Audio Open.")
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="Generate audio from a text prompt.")
    g.add_argument("prompt", nargs="?", default=None)
    g.add_argument("-o", "--output", default="output.wav")
    g.add_argument("--duration", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--guidance", type=float, default=7.0)
    g.add_argument("--negative-prompt", default=None)
    common_model_args(g)

    f = sub.add_parser("finetune", help="Train a LoRA adapter.")
    f.add_argument("dataset")
    f.add_argument("-o", "--output-dir", default="my_adapter")
    f.add_argument("--max-steps", type=int, default=200)
    f.add_argument("--rank", type=int, default=16)
    f.add_argument("--alpha", type=int, default=16)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--duration", type=float, default=10.0)
    f.add_argument("--seed", type=int, default=0)
    common_model_args(f)

    b = sub.add_parser("benchmark", help="Benchmark generation on this machine.")
    b.add_argument("--duration", type=float, default=10.0)
    b.add_argument("--steps", type=int, default=20)
    b.add_argument("--repeat", type=int, default=2)
    b.add_argument("--out", default=None)
    common_model_args(b)

    sub.add_parser("info", help="Show version, checkpoint, and preset table.")
    sub.add_parser("hardware", help="Show detected hardware and suggested memory mode.")
    sub.add_parser("presets", help="List memory presets and tradeoffs.")

    a = sub.add_parser("adapters", help="Inspect a saved adapter.")
    a.add_argument("adapter_dir")

    sub.add_parser("chat", help="Start the interactive session.")
    return p
