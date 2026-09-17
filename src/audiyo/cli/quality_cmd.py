from __future__ import annotations

import json
import os


def run_quality(args) -> int:
    from ..quality import analyze_file

    metrics = analyze_file(args.path)
    text = json.dumps(metrics, indent=2)
    if getattr(args, "out", None):
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("Wrote " + str(args.out))
    else:
        print(text)
    peak = float(metrics.get("peak", 0.0))
    clipped = float(metrics.get("clipped_fraction", 0.0))
    if peak < 0.02:
        print("Quiet file. Consider normalizing or checking generation settings.")
    if clipped > 0.01:
        print("Clipping detected. Lower guidance or enable the limiter.")
    return 0


def run_compare(args) -> int:
    from ..quality import compare_files

    report = compare_files(args.path_a, args.path_b)
    text = json.dumps(report, indent=2, default=str)
    if getattr(args, "out", None):
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("Wrote " + str(args.out))
    else:
        print(text)
    return 0


def run_sheet(args) -> int:
    from ..quality import write_listening_sheet

    folder = getattr(args, "folder", ".")
    out = getattr(args, "out", "sheet.csv")
    entries = []
    if os.path.isdir(folder):
        names = sorted(os.listdir(folder))
        for name in names:
            lower = name.lower()
            if lower.endswith((".wav", ".flac", ".ogg", ".opus")):
                entries.append({"file": os.path.join(folder, name), "prompt": "", "seed": "", "rating": "", "notes": ""})
    if not entries:
        print("No audio files found in " + str(folder) + ".")
        return 2
    write_listening_sheet(entries, out)
    print("Wrote " + str(out) + " with " + str(len(entries)) + " rows. Fill in rating and notes as you listen.")
    return 0
