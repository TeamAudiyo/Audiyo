from __future__ import annotations


def main():
    from audiyo import AudioModel

    for mode in ["performance", "balanced", "low", "minimal"]:
        model = AudioModel.__new__(AudioModel)
        model.memory_mode = mode
        from audiyo.memory import describe_presets

        info = describe_presets()[mode]
        print(mode + ": " + info["description"])


if __name__ == "__main__":
    main()
