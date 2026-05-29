"""Local OpenAI-compatible proxy for MacWhisper cloud transcription."""
from __future__ import annotations

import argparse
import sys
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from superwhisper_api.audio.models import audio_model
from superwhisper_api.auth import ensure_elevenlabs_key

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


class Segment(BaseModel):
    """One transcript segment in the verbose-JSON shape MacWhisper expects."""

    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0


class VerboseTranscription(BaseModel):
    """The OpenAI verbose-JSON transcription response MacWhisper consumes."""

    task: str = "transcribe"
    language: str = "unknown"
    duration: float = 0.0
    text: str = ""
    segments: list[Segment] = []


def _openai_verbose_response(data: dict[str, Any], language: str | None) -> VerboseTranscription:
    """Build the verbose transcription response MacWhisper accepts."""
    text = str(data.get("text") or "")
    duration = float(data.get("audio_duration_secs") or data.get("duration") or 0.0)
    language_code = data.get("language_code") or language or "unknown"

    segments: list[Segment] = []
    if text:
        segments.append(Segment(id=0, start=0.0, end=duration, text=text))

    return VerboseTranscription(
        language=language_code,
        duration=duration,
        text=text,
        segments=segments,
    )


def _require_token(authorization: str | None, expected_token: str | None) -> None:
    if expected_token is None:
        return
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Invalid proxy token.")


def create_app(*, proxy_token: str | None = None) -> FastAPI:
    """Create the MacWhisper proxy ASGI app."""
    app = FastAPI(title="Superwhisper MacWhisper Proxy")

    @app.get("/health")
    async def health() -> dict[str, bool]:
        return {"ok": True}

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = "scribe-v2",
        language: Annotated[str | None, Form()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        _require_token(authorization, proxy_token)
        try:
            spec = audio_model(model)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if spec.provider != "elevenlabs":
            raise HTTPException(
                status_code=400,
                detail=f"Model {model!r} is not supported by this proxy yet.",
            )

        audio = await file.read()
        files: dict[str, Any] = {
            "file": (file.filename or "speech.wav", audio, file.content_type or "audio/wav"),
            "model_id": (None, spec.model_id),
        }
        if language:
            files["language_code"] = (None, language)

        key = ensure_elevenlabs_key()
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                ELEVENLABS_URL,
                headers={"Accept": "application/json", "xi-api-key": key},
                files=files,
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return JSONResponse(_openai_verbose_response(response.json(), language).model_dump())

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MacWhisper transcription proxy.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--token",
        help="Optional bearer token MacWhisper must send as its Authentication Token.",
    )
    return parser


def main() -> int:
    """Run the MacWhisper proxy server (entry point: superwhisper-macwhisper-proxy)."""
    args = _build_parser().parse_args()
    app = create_app(proxy_token=args.token)
    print(f"MacWhisper proxy listening on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
