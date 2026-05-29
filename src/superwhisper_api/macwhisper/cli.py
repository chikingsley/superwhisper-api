"""CLI for the MacWhisper Global Replace helper: learn, apply."""
from __future__ import annotations

import argparse

from superwhisper_api.macwhisper.replacements import (
    add_apply_parser,
    add_learn_parser,
    add_remove_parser,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superwhisper-macwhisper",
        description="Grow MacWhisper's Global Replace dictionary with an agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_learn_parser(subparsers)
    add_apply_parser(subparsers)
    add_remove_parser(subparsers)
    return parser


def main() -> int:
    """Entry point for the superwhisper-macwhisper CLI."""
    args = _build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
