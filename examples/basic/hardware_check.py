from __future__ import annotations

import json


def main():
    from audiyo.hardware import detect_hardware, default_memory_mode

    hw = detect_hardware("auto")
    mode, why = default_memory_mode(hw.device, hw.gpu_vram_gb)
    print(json.dumps(hw.__dict__, indent=2, default=str))
    print("Suggested mode: " + mode)
    print(why)


if __name__ == "__main__":
    main()
