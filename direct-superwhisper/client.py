from __future__ import annotations

import json
import plistlib
import sqlite3
from pathlib import Path
from typing import Any

BASE_URL = "https://api.superwhisper.com"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/models/language/cloud"
CACHE_DB = Path.home() / "Library/Caches/com.superduper.superwhisper/Cache.db"
USER_AGENT = (
    "superwhisper/2.14.0 "
    "(com.superduper.superwhisper; build:2.14.0; macOS 26.5.0) "
    "Alamofire/5.8.0"
)

OPENAI_ALIASES = {
    "gpt-5.4-mini": "sw-gpt-5.4-mini",
    "gpt-5.4-nano": "sw-gpt-5.4-nano",
    "gpt-5.1": "sw-gpt-5.1",
    "gpt-5": "sw-gpt-5",
    "gpt-5-mini": "sw-gpt-5-mini",
    "gpt-5-nano": "sw-gpt-5-nano",
}
CACHE_COLUMNS = {"b.request_object", "d.receiver_data"}


class SuperwhisperCacheError(RuntimeError):
    """Raised when Superwhisper's local cache cannot provide proxy metadata."""


def _cache_row(request_url: str, column: str) -> bytes | None:
    if column not in CACHE_COLUMNS:
        raise ValueError(f"Invalid cache column: {column}")
    if not CACHE_DB.exists():
        return None
    with sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True) as conn:
        row = conn.execute(
            f"""
            select {column}
            from cfurl_cache_response r
            join cfurl_cache_blob_data b on b.entry_id = r.entry_id
            left join cfurl_cache_receiver_data d on d.entry_id = r.entry_id
            where r.request_key = ?
            order by r.time_stamp desc
            limit 1
            """,
            (request_url,),
        ).fetchone()
    if not row:
        return None
    value = row[0]
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode()
    return None


def signed_headers() -> dict[str, str]:
    data = _cache_row(BASE_URL + CHAT_COMPLETIONS_PATH, "b.request_object")
    if data is None:
        raise SuperwhisperCacheError("No cached Superwhisper chat request found.")
    request = plistlib.loads(data)["Array"]
    cached = request[19]
    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "user-agent": USER_AGENT,
    }
    for name in ("X-ID", "X-License", "X-Signature"):
        value = cached.get(name)
        if not isinstance(value, str) or not value:
            raise SuperwhisperCacheError(f"Cached Superwhisper request is missing {name}.")
        headers[name.lower()] = value
    return headers


def cached_models() -> list[dict[str, Any]]:
    data = _cache_row(BASE_URL + MODELS_PATH, "d.receiver_data")
    if data is None:
        raise SuperwhisperCacheError("No cached Superwhisper language model catalog found.")
    payload = json.loads(data)
    models = payload.get("models")
    if not isinstance(models, list):
        raise TypeError("Cached Superwhisper language model catalog is invalid.")
    return [model for model in models if isinstance(model, dict)]


def resolve_model(model: str) -> str:
    return OPENAI_ALIASES.get(model, model)


def route_for_model(model: str) -> tuple[str, str]:
    if model.startswith("claude-"):
        return "/anthropic/v1/messages", "anthropic"
    if model.startswith("gemini-"):
        return "/gemini/v1/messages", "gemini"
    if model.startswith("grok-"):
        return "/grok/v1/messages", "openai"
    return CHAT_COMPLETIONS_PATH, "openai"


def sse_json_lines(text: str) -> list[dict[str, Any]]:
    # The proxy currently streams SSE chunks even when the payload does not request streaming.
    events = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            continue
    return events


def _extract_anthropic_text(events: list[dict[str, Any]]) -> str:
    chunks = []
    for event in events:
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            chunks.append(delta["text"])
    return "".join(chunks)


def _extract_gemini_text(events: list[dict[str, Any]]) -> str:
    chunks = []
    for event in events:
        for candidate in event.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "".join(chunks)


def _extract_openai_text(events: list[dict[str, Any]], raw: dict[str, Any]) -> str:
    chunks = []
    source = events or [raw]
    for event in source:
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            if isinstance(delta.get("content"), str):
                chunks.append(delta["content"])
            elif isinstance(message.get("content"), str):
                chunks.append(message["content"])
    return "".join(chunks)


def output_text(format_name: str, events: list[dict[str, Any]], raw: dict[str, Any]) -> str:
    if format_name == "anthropic":
        return _extract_anthropic_text(events)
    if format_name == "gemini":
        return _extract_gemini_text(events)
    return _extract_openai_text(events, raw)


def returned_model(
    format_name: str, events: list[dict[str, Any]], raw: dict[str, Any]
) -> str | None:
    source = events or [raw]
    if format_name == "anthropic":
        for event in source:
            message = event.get("message")
            if isinstance(message, dict) and isinstance(message.get("model"), str):
                return message["model"]
    if format_name == "gemini":
        for event in source:
            if isinstance(event.get("modelVersion"), str):
                return event["modelVersion"]
    for event in source:
        if isinstance(event.get("model"), str):
            return event["model"]
    return None


def build_payload(
    format_name: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    system: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    use_max_completion_tokens: bool = False,
) -> dict[str, Any]:
    token_key = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
    if format_name in {"anthropic", "gemini"}:
        payload = {
            "model": model,
            token_key: max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if format_name == "anthropic" and system:
            payload["system"] = system
        return payload

    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    payload = {
        "model": model,
        token_key: max_tokens,
        "messages": messages,
    }
    if response_format:
        payload["response_format"] = response_format
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return payload
