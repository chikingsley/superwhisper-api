"""HTTP client for Superwhisper API text generation."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import httpx
from jsonschema import validate
from pydantic import BaseModel

from superwhisper_api.auth import cached_auth
from superwhisper_api.text.models import ModelSpec, model_spec

Message = Mapping[str, str]


class ModelResponse(BaseModel):
    """Structured response from a model generation call."""

    text: str
    model: str
    requested_model: str
    status_code: int
    events: list[dict[str, Any]]

    def parsed(self) -> Any:
        """Parse the response text as JSON (stripping any markdown fences)."""
        return parse_json_text(self.text)


class SuperwhisperClient:
    """Client for generating text via the Superwhisper API."""

    def __init__(
        self,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the client with optional base URL, headers, and HTTP client."""
        if base_url is None or headers is None:
            auth = cached_auth()
            base_url = auth.base_url if base_url is None else base_url
            headers = auth.headers if headers is None else headers
        self.base_url = base_url.rstrip("/")
        self.headers = headers
        self.http_client = http_client or httpx.Client(timeout=120)

    def generate(
        self,
        model: str | ModelSpec,
        messages: Iterable[Message],
        *,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Generate text from a model given a conversation."""
        spec = model_spec(model) if isinstance(model, str) else model
        payload: dict[str, Any] = {
            "model": spec.model_id,
            "max_tokens": max_tokens,
            "messages": list(messages),
        }
        if response_format and spec.supports_response_format:
            payload["response_format"] = response_format

        response = self.http_client.post(
            self.base_url + spec.path,
            headers=self.headers,
            json=payload,
        )
        response.raise_for_status()
        events = sse_events(response.text)
        return ModelResponse(
            text=model_text(spec.provider, events, response.text),
            model=returned_model(spec.provider, events) or "",
            requested_model=spec.model_id,
            status_code=response.status_code,
            events=events,
        )

    def generate_json(
        self,
        model: str | ModelSpec,
        messages: Iterable[Message],
        *,
        schema: dict[str, Any] | None = None,
        response_format_name: str = "structured_response",
        max_tokens: int = 512,
    ) -> Any:
        """Generate JSON-structured output from a model, optionally validating against a schema."""
        response_format = (
            json_schema_response_format(response_format_name, schema) if schema else None
        )
        response = self.generate(
            model,
            messages,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        parsed = response.parsed()
        if schema:
            validate(instance=parsed, schema=schema)
        return parsed


def json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI-compatible JSON schema response format dict."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def parse_json_text(text: str) -> Any:
    """Parse a string as JSON, stripping markdown code fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE event lines into a list of JSON event dicts."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            event = json.loads(line.removeprefix("data: "))
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


# --- Provider-specific SSE extractors -------------------------------------


def anthropic_text(events: list[dict[str, Any]]) -> str:
    """Concatenate text deltas from Anthropic streaming events."""
    chunks: list[str] = []
    for event in events:
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            chunks.append(delta["text"])
    return "".join(chunks)


def anthropic_model(events: list[dict[str, Any]]) -> str | None:
    """Return the model name from the first Anthropic message-start event."""
    for event in events:
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("model"), str):
            return message["model"]
    return None


def openai_text(events: list[dict[str, Any]]) -> str:
    """Concatenate content deltas from OpenAI-compatible streaming events."""
    chunks: list[str] = []
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                chunks.append(delta["content"])
    return "".join(chunks)


def openai_model(events: list[dict[str, Any]]) -> str | None:
    """Return the model name from the first OpenAI-compatible event that includes it."""
    for event in events:
        if isinstance(event.get("model"), str):
            return event["model"]
    return None


def gemini_text(events: list[dict[str, Any]]) -> str:
    """Concatenate text parts from Gemini streaming events."""
    chunks: list[str] = []
    for event in events:
        candidates = event.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "".join(chunks)


def gemini_model(events: list[dict[str, Any]]) -> str | None:
    """Return the model version from the first Gemini event that includes it."""
    for event in events:
        if isinstance(event.get("modelVersion"), str):
            return event["modelVersion"]
    return None


def model_text(provider: str, events: list[dict[str, Any]], raw_text: str) -> str:
    """Extract the model's text from a response (SSE stream or single JSON object).

    Superwhisper's proxy returns SSE for these routes most of the time, but
    intermittently returns a single non-streamed JSON object instead, so both
    shapes are handled for every provider.
    """
    if not events:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
        return text_from_object(provider, payload) if isinstance(payload, dict) else raw_text

    if provider == "anthropic":
        return anthropic_text(events)
    if provider == "gemini":
        return gemini_text(events)
    return openai_text(events)


def text_from_object(provider: str, payload: dict[str, Any]) -> str:
    """Extract text from a single non-streamed JSON response body."""
    if provider == "anthropic":
        blocks = payload.get("content")
        if isinstance(blocks, list):
            text = "".join(
                b["text"]
                for b in blocks
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            )
            if text:
                return text
    if provider == "gemini":
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if isinstance(parts, list):
                text = "".join(
                    p["text"]
                    for p in parts
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                )
                if text:
                    return text
    return text_from_chat_completion(payload)


def returned_model(provider: str, events: list[dict[str, Any]]) -> str | None:
    """Extract the returned model name from provider-specific SSE events."""
    if provider == "anthropic":
        return anthropic_model(events)
    if provider == "gemini":
        return gemini_model(events)
    return openai_model(events)


def text_from_chat_completion(payload: dict[str, Any]) -> str:
    """Extract message content from a chat completion payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""
