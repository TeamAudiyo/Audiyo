from __future__ import annotations

import sys


def main(argv=None) -> int:
    from .parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        from .chat import run_chat

        try:
            return run_chat(args)
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
    if args.command == "generate":
        from .generate_cmd import run_generate

        return run_generate(args)
    if args.command == "finetune":
        from .finetune_cmd import run_finetune

        return run_finetune(args)
    if args.command == "benchmark":
        from .info_cmd import run_bench

        return run_bench(args)
    if args.command == "info":
        from .info_cmd import run_info

        return run_info(args)
    if args.command == "hardware":
        from .info_cmd import run_hardware

        return run_hardware(args)
    if args.command == "presets":
        from .info_cmd import run_presets

        return run_presets(args)
    if args.command == "adapters":
        from .info_cmd import run_adapters

        return run_adapters(args)
    if args.command == "quality":
        from .quality_cmd import run_quality

        return run_quality(args)
    if args.command == "compare":
        from .quality_cmd import run_compare

        return run_compare(args)
    if args.command == "sheet":
        from .quality_cmd import run_sheet

        return run_sheet(args)
    if args.command == "estimate":
        from .system_cmd import run_estimate

        return run_estimate(args)
    if args.command == "ui":
        from .system_cmd import run_ui

        return run_ui(args)
    if args.command == "init-config":
        from .system_cmd import run_init_config

        return run_init_config(args)
    if args.command == "chat":
        from .chat import run_chat

        return run_chat(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
