from __future__ import annotations


def test_resolve_cli_wins_over_config():
    from audiyo.configfile import resolve_options

    defaults = {"duration": 10.0, "seed": 42, "memory_mode": "balanced"}
    cli = {"duration": 4.0, "seed": 42, "memory_mode": "balanced"}
    config = {"duration": 8.0, "seed": 7, "memory_mode": "low"}
    merged = resolve_options(defaults, cli, config)
    assert merged["duration"] == 4.0
    assert merged["seed"] == 7
    assert merged["memory_mode"] == "low"


def test_config_roundtrip(tmp_path):
    from audiyo.configfile import filter_generate_config, load_config_file, write_example_config

    path = str(tmp_path / "audiyo.json")
    assert write_example_config(path) == path
    data = load_config_file(path)
    assert data["prompt"].startswith("Rain")
    assert "duration" in filter_generate_config(data)


def test_config_rejects_missing(tmp_path):
    from audiyo.configfile import load_config_file
    from audiyo.errors import ValidationError

    try:
        load_config_file(str(tmp_path / "nope.json"))
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_estimate_stable_and_music():
    from audiyo.estimate import estimate_requirements

    stable = estimate_requirements("stabilityai/stable-audio-open-1.0", "balanced", gpu_vram_gb=8)
    assert stable["peak_gb"] > 1.0
    assert stable["verdict"] == "fits"
    music = estimate_requirements("MiniMaxAI/MiniMax-Music3", "balanced")
    assert music["peak_gb"] > 15.0
    assert music["backend"] == "minimax-music"


def test_estimate_rejects_unknown():
    from audiyo.errors import ValidationError
    from audiyo.estimate import estimate_requirements

    try:
        estimate_requirements("someone/else", "balanced")
    except ValidationError:
        return
    raise AssertionError("expected ValidationError")


def test_new_cli_commands_parse():
    from audiyo.cli.parser import build_parser

    parser = build_parser()
    assert parser.parse_args(["quality", "a.wav"]).command == "quality"
    assert parser.parse_args(["compare", "a.wav", "b.wav"]).command == "compare"
    assert parser.parse_args(["sheet", "."]).command == "sheet"
    assert parser.parse_args(["estimate"]).command == "estimate"
    assert parser.parse_args(["ui"]).command == "ui"
    assert parser.parse_args(["init-config"]).command == "init-config"
    args = parser.parse_args(["generate", "hello", "--fade-out-ms", "200", "--normalize-peak", "0.89"])
    assert args.fade_out_ms == 200.0
    assert args.normalize_peak == 0.89


def test_ui_module_imports_without_gradio():
    import audiyo.ui as ui

    assert callable(ui.launch_ui)
    assert callable(ui.build_demo)
