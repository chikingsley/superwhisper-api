"""Superwhisper text model specifications."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["openai", "anthropic", "gemini"]


@dataclass(frozen=True)
class ModelSpec:
    """Describes a Superwhisper text model and its API routing details."""

    key: str
    model_id: str
    path: str
    provider: Provider
    supports_response_format: bool = False
    observed_model: str | None = None


OPENAI_PATH = "/v1/chat/completions"
ANTHROPIC_PATH = "/anthropic/v1/messages"
GEMINI_PATH = "/gemini/v1/messages"

GPT_5_2 = ModelSpec(
    key="gpt-5.2",
    model_id="gpt-5.2",
    path=OPENAI_PATH,
    provider="openai",
    supports_response_format=True,
    observed_model="gpt-5.2-2025-12-11",
)
GPT_5_3_CHAT_LATEST = ModelSpec(
    key="gpt-5.3-chat-latest",
    model_id="gpt-5.3-chat-latest",
    path=OPENAI_PATH,
    provider="openai",
    supports_response_format=True,
    observed_model="gpt-5.3-chat-latest",
)
GPT_5_4_MINI = ModelSpec(
    key="gpt-5.4-mini",
    model_id="sw-gpt-5.4-mini",
    path=OPENAI_PATH,
    provider="openai",
    supports_response_format=True,
    observed_model="gpt-5.4-mini-2026-03-17",
)
GPT_5_4_NANO = ModelSpec(
    key="gpt-5.4-nano",
    model_id="sw-gpt-5.4-nano",
    path=OPENAI_PATH,
    provider="openai",
    supports_response_format=True,
    observed_model="gpt-5.4-nano-2026-03-17",
)
OPENAI_MODELS = (
    GPT_5_2,
    GPT_5_3_CHAT_LATEST,
    GPT_5_4_MINI,
    GPT_5_4_NANO,
)

SONNET_4_6 = ModelSpec(
    key="claude-sonnet-4-6",
    model_id="claude-sonnet-4-6",
    path=ANTHROPIC_PATH,
    provider="anthropic",
    observed_model="claude-sonnet-4-6",
)
HAIKU_4_5 = ModelSpec(
    key="claude-haiku-4-5",
    model_id="claude-haiku-4-5",
    path=ANTHROPIC_PATH,
    provider="anthropic",
    observed_model="claude-haiku-4-5-20251001",
)
ANTHROPIC_MODELS = (
    SONNET_4_6,
    HAIKU_4_5,
)

GEMINI_3_FLASH_PREVIEW = ModelSpec(
    key="gemini-3-flash-preview",
    model_id="gemini-3-flash-preview",
    path=GEMINI_PATH,
    provider="gemini",
    observed_model="gemini-3-flash-preview",
)
GEMINI_3_1_FLASH_LITE_PREVIEW = ModelSpec(
    key="gemini-3.1-flash-lite-preview",
    model_id="gemini-3.1-flash-lite-preview",
    path=GEMINI_PATH,
    provider="gemini",
    observed_model="gemini-3-flash-preview",
)
GEMINI_MODELS = (
    GEMINI_3_FLASH_PREVIEW,
    GEMINI_3_1_FLASH_LITE_PREVIEW,
)

SUPERWHISPER_MODELS = {
    model.key: model
    for model in (
        *OPENAI_MODELS,
        *ANTHROPIC_MODELS,
        *GEMINI_MODELS,
    )
}


def model_spec(name: str) -> ModelSpec:
    """Resolve a canonical text model key to its ModelSpec."""
    try:
        return SUPERWHISPER_MODELS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SUPERWHISPER_MODELS))
        raise ValueError(f"Unknown Superwhisper model {name!r}. Known models: {known}") from exc
