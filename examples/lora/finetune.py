from __future__ import annotations

import sys


def main():
    from audiyo import AudioModel

    dataset = sys.argv[1] if len(sys.argv) > 1 else "my_audio_dataset"
    out = sys.argv[2] if len(sys.argv) > 2 else "my_adapter"
    model = AudioModel.from_pretrained(device="auto", memory_mode="balanced")
    report = model.finetune(dataset=dataset, output_dir=out, max_steps=100)
    print("Done. Final loss " + str(round(report.final_loss, 4)))
    print("Checks " + str(report.checks))


if __name__ == "__main__":
    main()
