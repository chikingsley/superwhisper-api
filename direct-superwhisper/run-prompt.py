#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
from client import (
    BASE_URL,
    build_payload,
    output_text,
    resolve_model,
    returned_model,
    route_for_model,
    signed_headers,
    sse_json_lines,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call Superwhisper's signed language-model proxy directly."
    )
    parser.add_argument("--model", default="sw-gpt-5.4-mini")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--json", action="store_true", help="Print parsed metadata JSON.")
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument(
        "--json-object",
        action="store_true",
        help="Request JSON object mode on OpenAI-style routes.",
    )
    parser.add_argument(
        "--schema-file",
        help="JSON file containing a response_format object to pass through.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Pass reasoning_effort on OpenAI-style routes.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        action="store_true",
        help="Use max_completion_tokens instead of max_tokens.",
    )
    parser.add_argument("prompt", nargs="*", help="Prompt text. Reads stdin if omitted.")
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or sys.stdin.read()
    model = resolve_model(args.model)
    endpoint, format_name = route_for_model(model)
    response_format = None
    if args.json_object:
        response_format = {"type": "json_object"}
    if args.schema_file:
        with Path(args.schema_file).open(encoding="utf-8") as handle:
            response_format = json.load(handle)
    payload = build_payload(
        format_name,
        model,
        prompt,
        args.max_tokens,
        system=args.system,
        response_format=response_format,
        reasoning_effort=args.reasoning_effort,
        use_max_completion_tokens=args.max_completion_tokens,
    )

    response = httpx.post(
        BASE_URL + endpoint,
        headers=signed_headers(),
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    text = response.text
    events = sse_json_lines(text)
    try:
        raw = response.json()
    except json.JSONDecodeError:
        raw = {}

    result = {
        "requested_model": args.model,
        "sent_model": model,
        "returned_model": returned_model(format_name, events, raw),
        "endpoint": endpoint,
        "text": output_text(format_name, events, raw),
    }
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
