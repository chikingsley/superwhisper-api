#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

from __future__ import annotations

import argparse
import json
import sys

import httpx
from client import (
    BASE_URL,
    SuperwhisperCacheError,
    build_payload,
    cached_models,
    output_text,
    resolve_model,
    returned_model,
    route_for_model,
    signed_headers,
    sse_json_lines,
)

DEFAULT_MODELS = [
    "gpt-5.2",
    "gpt-5.3-chat-latest",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "grok-4-1-fast-non-reasoning",
    "gpt-5.5-low",
    "gpt-5.3-codex-spark",
]


def probe(model: str) -> dict[str, object]:
    sent_model = resolve_model(model)
    endpoint, format_name = route_for_model(sent_model)
    payload = build_payload(format_name, sent_model, "reply exactly ok", 16)
    try:
        response = httpx.post(
            BASE_URL + endpoint,
            headers=signed_headers(),
            json=payload,
            timeout=60,
        )
        events = sse_json_lines(response.text)
        try:
            raw = response.json()
        except json.JSONDecodeError:
            raw = {}
        model_back = returned_model(format_name, events, raw)
        text = output_text(format_name, events, raw)
        fallback = model_back == "gpt-3.5-turbo-0125"
        expected = sent_model.removeprefix("sw-")
        model_mismatch = (
            isinstance(model_back, str)
            and not model_back.startswith(expected)
            and not fallback
        )
        return {
            "requested_model": model,
            "sent_model": sent_model,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "returned_model": model_back,
            "ok": response.is_success and not fallback and not model_mismatch,
            "fallback": fallback,
            "model_mismatch": model_mismatch,
            "text": text[:120],
        }
    except (
        httpx.RequestError,
        json.JSONDecodeError,
        OSError,
        SuperwhisperCacheError,
        TypeError,
        ValueError,
    ) as exc:
        return {
            "requested_model": model,
            "sent_model": sent_model,
            "endpoint": endpoint,
            "status_code": None,
            "returned_model": None,
            "ok": False,
            "fallback": False,
            "model_mismatch": False,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Superwhisper proxy model IDs.")
    parser.add_argument("models", nargs="*")
    parser.add_argument("--catalog", action="store_true", help="Print cached catalog first.")
    args = parser.parse_args()

    if args.catalog:
        for model in cached_models():
            print(
                json.dumps(
                    {
                        "id": model.get("id"),
                        "name": model.get("name"),
                        "format": model.get("format"),
                        "completionPath": model.get("completionPath"),
                        "deprecated": model.get("deprecated"),
                    },
                    ensure_ascii=False,
                )
            )

    models = args.models or DEFAULT_MODELS
    ok = True
    for model in models:
        result = probe(model)
        print(json.dumps(result, ensure_ascii=False))
        ok = ok and bool(result["ok"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
