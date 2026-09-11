from __future__ import annotations


def test_parser_builds():
    from audiyo.cli.parser import build_parser

    p = build_parser()
    args = p.parse_args(["info"])
    assert args.command == "info"
    args = p.parse_args(["generate", "hello", "--duration", "4"])
    assert args.prompt == "hello"
    assert args.duration == 4.0


def test_info_runs():
    from types import SimpleNamespace

    from audiyo.cli.info_cmd import run_info

    assert run_info(SimpleNamespace()) == 0


def test_presets_runs():
    from types import SimpleNamespace

    from audiyo.cli.info_cmd import run_presets

    assert run_presets(SimpleNamespace()) == 0


def test_hardware_runs():
    from types import SimpleNamespace

    from audiyo.cli.info_cmd import run_hardware

    assert run_hardware(SimpleNamespace()) == 0


def test_ask_helpers():
    import builtins

    from audiyo.cli.chat import ask_float, ask_int

    real = builtins.input
    builtins.input = lambda *a, **k: "abc"
    try:
        calls = {"n": 0}

        def fake(prompt=""):
            calls["n"] += 1
            if calls["n"] < 2:
                return "abc"
            return "7"

        builtins.input = fake
        assert ask_int("Seed", 1) == 7
    finally:
        builtins.input = real


def test_main_info_dispatch():
    from audiyo.cli.main import main

    assert main(["info"]) == 0
    assert main(["presets"]) == 0
    assert main(["hardware"]) == 0
