"""Shared transcription orchestration plus per-provider request logic."""
from __future__ import annotations

import base64
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from superwhisper_api.audio.models import AudioModelSpec, audio_model
from superwhisper_api.auth import (
    cached_auth,
    ensure_elevenlabs_key,
    ensure_v1_key,
    ensure_v2_key,
)

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEEPGRAM_PROXY_URL = "https://api.superwhisper.com/deepgram/v1/listen"
S1_URL = "https://us.ai.superwhisper.com/generate"
ULTRA_URL = "https://ai.superwhisper.com/v1/c/run"


class TranscriptResult(Protocol):
    """Common interface for transcript and failure objects from all providers."""

    audio_path: str

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary representation of the result."""
        ...


def superwhisper_datetime() -> str:
    """Return the current UTC datetime in Superwhisper's timestamp format."""
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


@dataclass(frozen=True)
class Transcript:
    """Normalized transcription result plus the raw provider response."""

    audio_path: str
    provider: str
    model_key: str
    model_id: str
    transcript: str
    raw_response: dict[str, Any]
    recording_id: str = ""
    duration: float | None = None
    processing_time: int | None = None
    created_at: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary representation of the transcript."""
        return {
            "audio_path": self.audio_path,
            "provider": self.provider,
            "model_key": self.model_key,
            "model_id": self.model_id,
            "recording_id": self.recording_id,
            "datetime": self.created_at or superwhisper_datetime(),
            "duration": self.duration,
            "processing_time": self.processing_time,
            "transcript": self.transcript,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class Failure:
    """Failed transcription attempt metadata."""

    audio_path: str
    error: str
    attempts: int = 1
    created_at: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a dictionary representation of the failure."""
        return {
            "audio_path": self.audio_path,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at or datetime.now(UTC).isoformat(),
        }


def failure_from_exception(audio_path: str, exc: Exception) -> Failure:
    """Build a Failure from an exception raised while transcribing."""
    return Failure(
        audio_path=audio_path,
        error=str(exc),
        attempts=1,
        created_at=datetime.now(UTC).isoformat(),
    )


ExtractFn = Callable[[dict[str, Any]], "str | float | None"]
KeyFn = Callable[[], str]
ProcessFn = Callable[[Path], TranscriptResult]


def transcribe_file(
    audio: Path,
    *,
    provider: str,
    model: str,
    language: str | None,
    transcribe_raw: Callable[..., dict[str, Any]],
    extract_transcript: ExtractFn,
    extract_duration: ExtractFn,
    extract_recording_id: ExtractFn,
    api_key: str | None = None,
    ensure_key: KeyFn | None = None,
) -> TranscriptResult:
    """Transcribe audio through a provider, handling timing and error wrapping."""
    started = time.monotonic()
    try:
        key = api_key or (ensure_key() if ensure_key else None)
        spec = audio_model(model)

        if key:
            data = transcribe_raw(audio, key, model=model, language=language)
        else:
            data = transcribe_raw(audio, model=model, language=language)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        return Transcript(
            audio_path=str(audio),
            provider=provider,
            model_key=spec.key,
            model_id=spec.model_id,
            recording_id=str(extract_recording_id(data) or ""),
            created_at=superwhisper_datetime(),
            duration=extract_duration(data),
            processing_time=elapsed_ms,
            transcript=extract_transcript(data) or "",
            raw_response=data,
        )
    except Exception as exc:
        return failure_from_exception(str(audio), exc)


# --- ElevenLabs -----------------------------------------------------------


def _elevenlabs_raw(
    audio: Path,
    api_key: str,
    *,
    model: str = "scribe-v2",
    language: str | None = None,
) -> dict[str, Any]:
    model_id = audio_model(model).model_id
    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "xi-api-key": api_key,
    }
    with audio.open("rb") as handle:
        files: Any = {
            "file": (audio.name, handle, "audio/wav"),
            "model_id": (None, model_id),
        }
        if language:
            files["language_code"] = (None, language)
        response = httpx.post(ELEVENLABS_URL, headers=headers, files=files, timeout=600)
        response.raise_for_status()
        return response.json()


def _elevenlabs_transcript(data: dict[str, Any]) -> str | None:
    return data.get("text")


def _elevenlabs_duration(data: dict[str, Any]) -> float | None:
    duration = data.get("duration")
    return float(duration) if isinstance(duration, int | float) else None


def _elevenlabs_recording_id(data: dict[str, Any]) -> str | None:
    return data.get("transcription_id")


# --- Deepgram (via Superwhisper proxy) ------------------------------------


def _deepgram_raw(
    audio: Path,
    *,
    model: str = "deepgram-nova-2",
    language: str | None = None,
) -> dict[str, Any]:
    model_id = audio_model(model).model_id
    headers = {
        **cached_auth().headers,
        "Accept": "application/json",
        "Content-Type": "audio/wav",
    }
    params: dict[str, str] = {"model": model_id}
    if language:
        params["language"] = language
    with audio.open("rb") as handle:
        response = httpx.post(
            DEEPGRAM_PROXY_URL,
            headers=headers,
            params=params,
            content=handle.read(),
            timeout=600,
        )
        response.raise_for_status()
        return response.json()


def _deepgram_transcript(data: dict[str, Any]) -> str | None:
    channels = data.get("results", {}).get("channels", [])
    if not channels:
        return None
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return None
    return str(alternatives[0].get("transcript", ""))


def _deepgram_duration(data: dict[str, Any]) -> float | None:
    duration = data.get("metadata", {}).get("duration")
    return float(duration) if isinstance(duration, int | float) else None


def _deepgram_recording_id(data: dict[str, Any]) -> str | None:
    return data.get("request_id")


# --- Superwhisper S1 Voice / Ultra (base64 JSON) --------------------------


def _segments_text(segments: list) -> str | None:
    parts = [str(seg.get("text", "")) for seg in segments if seg.get("text")]
    return " ".join(parts) or None


def _s1_raw(
    audio: Path,
    api_key: str,
    *,
    model: str = "s1-voice",
    language: str | None = None,
) -> dict[str, Any]:
    audio_model(model)
    with audio.open("rb") as handle:
        audio_b64 = base64.b64encode(handle.read()).decode()
    payload = {"audio_base64": audio_b64, "language": language or "en"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = httpx.post(S1_URL, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _s1_transcript(data: dict[str, Any]) -> str | None:
    text = data.get("transcription") or data.get("text") or ""
    if text:
        return str(text)
    return _segments_text(data.get("segments", []))


def _s1_duration(data: dict[str, Any]) -> float | None:
    duration = data.get("file_duration")
    return float(duration) if isinstance(duration, int | float) else None


def _s1_recording_id(data: dict[str, Any]) -> str | None:
    return data.get("request_id")


def _ultra_raw(
    audio: Path,
    api_key: str,
    *,
    model: str = "ultra",
    language: str | None = None,
) -> dict[str, Any]:
    audio_model(model)
    with audio.open("rb") as handle:
        audio_b64 = base64.b64encode(handle.read()).decode()
    payload: dict[str, object] = {
        "audio_base64": audio_b64,
        "word_timestamps": False,
        "translate": False,
    }
    if language:
        payload["language"] = language
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    response = httpx.post(ULTRA_URL, headers=headers, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def _ultra_transcript(data: dict[str, Any]) -> str | None:
    result = data.get("result", {})
    text = result.get("text") or ""
    if text:
        return str(text)
    return _segments_text(result.get("segments", []))


def _ultra_duration(data: dict[str, Any]) -> float | None:
    duration = data.get("result", {}).get("file_duration")
    return float(duration) if isinstance(duration, int | float) else None


def _ultra_recording_id(data: dict[str, Any]) -> str | None:
    return data.get("run_id")


@dataclass(frozen=True)
class _Provider:
    """How to call one audio provider and pull fields from its response."""

    transcribe_raw: Callable[..., dict[str, Any]]
    extract_transcript: ExtractFn
    extract_duration: ExtractFn
    extract_recording_id: ExtractFn
    ensure_key: KeyFn | None = None


PROVIDERS: dict[str, _Provider] = {
    "elevenlabs": _Provider(
        _elevenlabs_raw,
        _elevenlabs_transcript,
        _elevenlabs_duration,
        _elevenlabs_recording_id,
        ensure_key=ensure_elevenlabs_key,
    ),
    "deepgram": _Provider(
        _deepgram_raw,
        _deepgram_transcript,
        _deepgram_duration,
        _deepgram_recording_id,
    ),
    "s1": _Provider(
        _s1_raw,
        _s1_transcript,
        _s1_duration,
        _s1_recording_id,
        ensure_key=ensure_v2_key,
    ),
    "ultra": _Provider(
        _ultra_raw,
        _ultra_transcript,
        _ultra_duration,
        _ultra_recording_id,
        ensure_key=ensure_v1_key,
    ),
}


def warn_if_key_ignored(provider: str, key: str | None) -> None:
    """Print a warning when --key is supplied for a provider that ignores it."""
    if not key:
        return
    messages = {
        "deepgram": "Warning: --key is ignored for Deepgram (uses Superwhisper proxy auth)",
        "s1": "Warning: --key is ignored for S1 Voice (uses v2 inference key)",
        "ultra": "Warning: --key is ignored for Ultra (uses v1 inference key)",
    }
    msg = messages.get(provider)
    if msg:
        print(msg, file=sys.stderr)


def create_process_fn(
    spec: AudioModelSpec,
    key: str | None,
    language: str | None = None,
) -> ProcessFn:
    """Return a transcription function bound to the provider, key, and language."""
    provider = PROVIDERS.get(spec.provider)
    if provider is None:
        raise ValueError(f"Unsupported provider '{spec.provider}' for model '{spec.key}'")

    def process(audio: Path) -> TranscriptResult:
        return transcribe_file(
            audio,
            provider=spec.provider,
            model=spec.key,
            language=language,
            transcribe_raw=provider.transcribe_raw,
            extract_transcript=provider.extract_transcript,
            extract_duration=provider.extract_duration,
            extract_recording_id=provider.extract_recording_id,
            api_key=key,
            ensure_key=provider.ensure_key,
        )

    return process
