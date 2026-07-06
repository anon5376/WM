from __future__ import annotations

import argparse
import sys

from wm.generate import main as generate_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="wm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", required=True)
    generate_parser.add_argument("--checkpoint", required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-tokens", type=int, default=64)
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        sys.argv = [
            "wm.generate",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--temperature",
            str(args.temperature),
        ]
        if args.json:
            sys.argv.append("--json")
        generate_main()
