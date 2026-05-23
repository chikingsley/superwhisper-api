#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from _key import ensure_key

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"


@dataclass
class Transcript:
    audio_path: str
    transcription_id: str | None
    text: str
    language_code: str
    language_probability: float
    duration: float | None
    model: str
    error: str | None
    created_at: str


def transcribe_raw(
    audio: Path,
    api_key: str,
    *,
    model: str = "scribe_v2",
    language: str | None = None,
) -> dict:
    headers = {
        "User-Agent": "superwhisper/2.14.0 CFNetwork/1568.100.1 Darwin/25.5.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "xi-api-key": api_key,
    }
    with audio.open("rb") as f:
        files: dict = {
            "file": (audio.name, f, "audio/wav"),
            "model_id": (None, model),
        }
        if language:
            files["language_code"] = (None, language)

        response = httpx.post(
            ELEVENLABS_URL,
            headers=headers,
            files=files,
            timeout=600,
        )
        response.raise_for_status()
        return response.json()


def transcribe_file_batch(
    audio: Path,
    api_key: str,
    *,
    model: str = "scribe_v2",
    language: str | None = None,
) -> Transcript:
    started = time.monotonic()
    try:
        data = transcribe_raw(audio, api_key, model=model, language=language)
        return Transcript(
            audio_path=str(audio),
            transcription_id=data.get("transcription_id"),
            text=data.get("text", ""),
            language_code=data.get("language_code", ""),
            language_probability=data.get("language_probability", 0.0),
            duration=time.monotonic() - started,
            model=model,
            error=None,
            created_at=datetime.now(UTC).isoformat(),
        )
    except Exception as exc:
        return Transcript(
            audio_path=str(audio),
            transcription_id=None,
            text="",
            language_code="",
            language_probability=0.0,
            duration=time.monotonic() - started,
            model=model,
            error=str(exc),
            created_at=datetime.now(UTC).isoformat(),
        )


def last_written(path: Path) -> str | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(audio := payload.get("audio_path"), str):
                last = audio
    return last


def iter_pending(paths_file: Path, jsonl: Path) -> list[Path]:
    resume = last_written(jsonl)
    found = resume is None
    pending: list[Path] = []
    skipped = 0
    with paths_file.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not found:
                skipped += 1
                if stripped == resume:
                    found = True
                    print(f"Resuming after {skipped} done paths", file=sys.stderr)
                continue
            pending.append(Path(stripped))
    if not found and resume is not None:
        print("Resume path not found; no new paths", file=sys.stderr)
    return pending


def write_jsonl(path: Path, t: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Transcribe audio via ElevenLabs API (Single or Batch Mode)"
    )
    # Target options
    p.add_argument(
        "audio",
        nargs="?",
        type=Path,
        help="Path to a single audio file to transcribe (omit if using --paths-file)",
    )
    p.add_argument(
        "--paths-file",
        type=Path,
        help="Path to manifest file with list of audio paths (one per line) for batch processing",
    )

    # Output options
    p.add_argument(
        "--jsonl",
        type=Path,
        help="Output JSONL file path for batch results (required in batch mode)",
    )
    p.add_argument(
        "--fail-jsonl",
        type=Path,
        help="Output JSONL file path for batch failures (optional)",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (single-file mode only)",
    )

    # Common API settings
    p.add_argument(
        "--key",
        help="ElevenLabs API key (auto-extracted from Superwhisper if omitted)",
    )
    p.add_argument("--model", default="scribe_v2", help="Model ID (default: scribe_v2)")
    p.add_argument("--language", help="Language code hint (e.g. fas, eng)")
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers in batch mode (default: 4)",
    )

    args = p.parse_args()

    # Input validation
    if not args.audio and not args.paths_file:
        p.error("Either a single 'audio' file or '--paths-file' must be specified.")
    if args.audio and args.paths_file:
        p.error("Cannot specify both a single 'audio' file and '--paths-file'.")
    if args.paths_file and not args.jsonl:
        p.error("--jsonl must be specified when using --paths-file.")

    # API key retrieval
    api_key = args.key or ensure_key()

    # Single file mode
    if args.audio:
        if not args.audio.exists():
            print(f"File not found: {args.audio}", file=sys.stderr)
            sys.exit(1)

        try:
            result = transcribe_raw(
                args.audio,
                api_key,
                model=args.model,
                language=args.language,
            )
            indent = 2 if args.pretty else None
            print(json.dumps(result, indent=indent, ensure_ascii=False))
        except Exception as exc:
            print(f"Error transcribing file: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    if not args.paths_file.exists():
        print(f"Paths file not found: {args.paths_file}", file=sys.stderr)
        sys.exit(1)

    pending = iter_pending(args.paths_file, args.jsonl)
    if not pending:
        print("No pending audio files.", file=sys.stderr)
        return

    print(f"Processing {len(pending)} files with model={args.model}", file=sys.stderr)

    def process(audio: Path) -> Transcript:
        return transcribe_file_batch(audio, api_key, model=args.model, language=args.language)

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process, audio): audio for audio in pending}
        for fut in as_completed(futures):
            t = fut.result()
            if t.error:
                fail += 1
                if args.fail_jsonl:
                    write_jsonl(args.fail_jsonl, t)
                print(f"FAIL {t.audio_path}: {t.error}", file=sys.stderr)
            else:
                ok += 1
                write_jsonl(args.jsonl, t)
                snippet = t.text[:80].replace("\n", " ")
                print(f"OK {t.audio_path}: {snippet}", file=sys.stderr)

    print(f"Done: {ok} ok, {fail} failed", file=sys.stderr)


if __name__ == "__main__":
    main()
