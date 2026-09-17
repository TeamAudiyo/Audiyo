from __future__ import annotations

import argparse

from ..config import SUPPORTED_CHECKPOINT


def common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", default=SUPPORTED_CHECKPOINT)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--memory-mode", default="balanced", choices=["performance", "balanced", "low", "minimal"])
    p.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"])
    p.add_argument("--token", default=None)
    p.add_argument("--config", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiyo", description="Memory-efficient text-to-audio with Stable Audio Open.")
    sub = p.add_subparsers(dest="command")

    g = sub.add_parser("generate", help="Generate audio from a text prompt.")
    g.add_argument("prompt", nargs="?", default=None)
    g.add_argument("-o", "--output", default="output.wav")
    g.add_argument("--duration", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--seeds", default=None)
    g.add_argument("--steps", type=int, default=100)
    g.add_argument("--guidance", type=float, default=7.0)
    g.add_argument("--negative-prompt", default=None)
    g.add_argument("--lyrics", default=None)
    g.add_argument("--fade-in-ms", type=float, default=0.0)
    g.add_argument("--fade-out-ms", type=float, default=0.0)
    g.add_argument("--normalize-peak", type=float, default=None)
    g.add_argument("--limiter", action="store_true")
    g.add_argument("--trim-silence", action="store_true")
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

    q = sub.add_parser("quality", help="Measure levels and spectrum of an audio file.")
    q.add_argument("path")
    q.add_argument("--out", default=None)

    c = sub.add_parser("compare", help="Compare two audio files.")
    c.add_argument("path_a")
    c.add_argument("path_b")
    c.add_argument("--out", default=None)

    s = sub.add_parser("sheet", help="Write a listening sheet for a folder of audio.")
    s.add_argument("folder", nargs="?", default=".")
    s.add_argument("--out", default="sheet.csv")

    e = sub.add_parser("estimate", help="Estimate memory needs before loading.")
    e.add_argument("--checkpoint", default=SUPPORTED_CHECKPOINT)
    e.add_argument("--memory-mode", default="balanced", choices=["performance", "balanced", "low", "minimal"])
    e.add_argument("--vram", type=float, default=None)

    u = sub.add_parser("ui", help="Open a local web interface.")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true")
    u.add_argument("--checkpoint", default=SUPPORTED_CHECKPOINT)

    n = sub.add_parser("init-config", help="Write an example generate config file.")
    n.add_argument("--out", default="audiyo.json")

    sub.add_parser("chat", help="Start the interactive session.")
    return p
