from __future__ import annotations


def ask(prompt: str, default: str = "") -> str:
    suffix = " [" + default + "]" if default else ""
    try:
        value = input(prompt + suffix + ": ").strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        raise
    if not value:
        return default
    return value


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = ask(prompt, str(default))
        try:
            return float(raw)
        except ValueError:
            print("That is not a number, try again.")


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print("That is not a whole number, try again.")


def run_chat(args) -> int:
    print("Welcome to Audiyo. Type the number of what you want to do.")
    model = None
    while True:
        print("")
        print("1 Generate audio")
        print("2 Train a LoRA adapter")
        print("3 Show hardware and suggested memory mode")
        print("4 List memory presets")
        print("5 Benchmark this machine")
        print("6 Quit")
        try:
            choice = input("Choice [6]: ").strip() or "6"
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if choice == "6" or choice.lower() in ("quit", "exit", "q"):
            print("Bye.")
            return 0
        elif choice == "1":
            model = chat_generate(model)
        elif choice == "2":
            model = chat_finetune(model)
        elif choice == "3":
            from .info_cmd import run_hardware

            run_hardware(args)
        elif choice == "4":
            from .info_cmd import run_presets

            run_presets(args)
        elif choice == "5":
            chat_benchmark(model)
        else:
            print("Unknown choice, pick 1 to 6.")


def get_model(model, args):
    if model is not None:
        return model
    from ..model import AudioModel

    mode = ask("Memory mode", "balanced")
    if mode not in ("performance", "balanced", "low", "minimal"):
        print("Unknown mode, using balanced.")
        mode = "balanced"
    print("Loading the model, this takes a while the first time.")
    model = AudioModel.from_pretrained(device="auto", memory_mode=mode)
    print("Model ready.")
    return model


def chat_generate(model, args=None):
    from types import SimpleNamespace

    from .generate_cmd import run_generate

    prompt = ask("Prompt", "Rain against a window with distant thunder")
    duration = ask_float("Duration in seconds", 10.0)
    seed = ask_int("Seed", 42)
    steps = ask_int("Inference steps", 50)
    output = ask("Output file", "output.wav")
    fake = SimpleNamespace(
        prompt=prompt,
        output=output,
        duration=duration,
        seed=seed,
        steps=steps,
        guidance=7.0,
        negative_prompt=None,
        checkpoint="stabilityai/stable-audio-open-1.0",
        device="auto",
        memory_mode="balanced",
        dtype=None,
        token=None,
    )
    try:
        code = run_generate(fake)
    except (EOFError, KeyboardInterrupt):
        print("")
        return model
    except Exception as exc:
        print("Generation failed: " + str(exc)[:500])
        return model
    if code == 0:
        print("Done.")
    return model


def chat_finetune(model, args=None):
    from types import SimpleNamespace

    from .finetune_cmd import run_finetune

    dataset = ask("Dataset folder or CSV", "my_data")
    out = ask("Output directory", "my_adapter")
    steps = ask_int("Max steps", 200)
    fake = SimpleNamespace(
        dataset=dataset,
        output_dir=out,
        max_steps=steps,
        rank=16,
        alpha=16,
        lr=1e-4,
        duration=10.0,
        seed=0,
        checkpoint="stabilityai/stable-audio-open-1.0",
        device="auto",
        memory_mode="balanced",
        dtype=None,
        token=None,
    )
    try:
        run_finetune(fake)
    except (EOFError, KeyboardInterrupt):
        print("")
        return model
    except Exception as exc:
        print("Training failed: " + str(exc)[:500])
    return model


def chat_benchmark(model, args=None):
    print("Running a short benchmark, this takes a few minutes on GPU.")
    from types import SimpleNamespace

    from .info_cmd import run_bench

    fake = SimpleNamespace(
        duration=4.0,
        steps=20,
        repeat=2,
        out=None,
        checkpoint="stabilityai/stable-audio-open-1.0",
        device="auto",
        memory_mode="balanced",
        dtype=None,
        token=None,
    )
    try:
        run_bench(fake)
    except (EOFError, KeyboardInterrupt):
        print("")
    except Exception as exc:
        print("Benchmark failed: " + str(exc)[:500])
