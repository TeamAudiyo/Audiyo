from __future__ import annotations

import json


def main():
    import torch

    from audiyo.hardware import detect_hardware

    hw = detect_hardware("auto")
    print("cuda: " + str(hw.cuda_available))
    print("device: " + hw.device)
    print("torch: " + hw.torch_version)
    try:
        import diffusers
        import transformers
        import peft

        print("diffusers: " + diffusers.__version__)
        print("transformers: " + transformers.__version__)
        print("peft: " + peft.__version__)
    except Exception as exc:
        print("optional dep missing: " + str(exc))
    print("torch cuda build: " + str(torch.version.cuda))
    print(json.dumps(hw.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
