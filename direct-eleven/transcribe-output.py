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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from _key import ensure_key

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"


@dataclass
class Transcript:
    audio_path: str
    recording_rowid: int | None
    recording_id: str
    datetime: str
    folder_name: str
    model_name: str
    mode_name: str
    duration: float | None
    processing_time: int | None
    transcript: str

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "recording_rowid": self.recording_rowid,
            "recording_id": self.recording_id,
            "datetime": self.datetime,
            "folder_name": self.folder_name,
            "model_name": self.model_name,
            "mode_name": self.mode_name,
            "duration": self.duration,
            "processing_time": self.processing_time,
            "transcript": self.transcript,
        }


@dataclass
class Failure:
    audio_path: str
    error: str
    attempts: int
    created_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "audio_path": self.audio_path,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
        }


def superwhisper_datetime() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def model_name(model: str) -> str:
    if model == "scribe_v2":
        return "Scribe (Cloud)"
    return model


def duration_from_response(data: dict) -> float | None:
    duration = data.get("duration")
    if isinstance(duration, int | float):
        return float(duration)
    return None


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
) -> Transcript | Failure:
    started = time.monotonic()
    try:
        data = transcribe_raw(audio, api_key, model=model, language=language)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return Transcript(
            audio_path=str(audio),
            recording_rowid=None,
            recording_id=str(data.get("transcription_id") or ""),
            datetime=superwhisper_datetime(),
            folder_name="direct-eleven",
            model_name=model_name(model),
            mode_name="direct-eleven",
            duration=duration_from_response(data),
            processing_time=elapsed_ms,
            transcript=str(data.get("text") or ""),
        )
    except Exception as exc:
        return Failure(
            audio_path=str(audio),
            error=str(exc),
            attempts=1,
            created_at=datetime.now(UTC).isoformat(),
        )


def written_audio_paths(path: Path) -> set[str]:
    paths: set[str] = set()
    if not path.exists():
        return paths
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(audio := payload.get("audio_path"), str):
                paths.add(audio)
    return paths


def iter_pending(paths_file: Path, jsonl: Path) -> Iterator[Path]:
    completed = written_audio_paths(jsonl)
    skipped = 0

    def pending_paths() -> Iterator[Path]:
        nonlocal skipped
        try:
            with paths_file.open(encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped in completed:
                        skipped += 1
                        continue
                    yield Path(stripped)
        finally:
            if skipped:
                print(f"Skipping {skipped} already-written paths", file=sys.stderr)

    return pending_paths()


def write_jsonl(path: Path, t: Transcript | Failure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(t.as_dict(), ensure_ascii=False) + "\n")


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
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")

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
            result = transcribe_file_batch(
                args.audio,
                api_key,
                model=args.model,
                language=args.language,
            )
            indent = 2 if args.pretty else None
            print(json.dumps(result.as_dict(), indent=indent, ensure_ascii=False))
            if isinstance(result, Failure):
                sys.exit(1)
        except Exception as exc:
            print(f"Error transcribing file: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode
    if not args.paths_file.exists():
        print(f"Paths file not found: {args.paths_file}", file=sys.stderr)
        sys.exit(1)

    pending = iter_pending(args.paths_file, args.jsonl)
    print(f"Processing files with model={args.model}", file=sys.stderr)

    def process(audio: Path) -> Transcript | Failure:
        return transcribe_file_batch(audio, api_key, model=args.model, language=args.language)

    ok = 0
    fail = 0
    submitted = 0
    max_in_flight = max(args.max_workers * 4, args.max_workers)

    def submit_next(
        pool: ThreadPoolExecutor,
        pending_paths: Iterator[Path],
        futures: set[Future[Transcript | Failure]],
    ) -> bool:
        nonlocal submitted
        try:
            audio = next(pending_paths)
        except StopIteration:
            return False
        futures.add(pool.submit(process, audio))
        submitted += 1
        return True

    def handle_done(done: set[Future[Transcript | Failure]]) -> None:
        nonlocal ok, fail
        for fut in done:
            t = fut.result()
            if isinstance(t, Failure):
                fail += 1
                if args.fail_jsonl:
                    write_jsonl(args.fail_jsonl, t)
                print(f"FAIL {t.audio_path}: {t.error}", file=sys.stderr)
            else:
                ok += 1
                write_jsonl(args.jsonl, t)
                snippet = t.transcript[:80].replace("\n", " ")
                print(f"OK {t.audio_path}: {snippet}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures: set[Future[Transcript | Failure]] = set()
        while len(futures) < max_in_flight and submit_next(pool, pending, futures):
            pass
        if not futures:
            print("No pending audio files.", file=sys.stderr)
            return

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            handle_done(done)
            while len(futures) < max_in_flight and submit_next(pool, pending, futures):
                pass

    print(f"Done: {ok} ok, {fail} failed, {submitted} submitted", file=sys.stderr)


if __name__ == "__main__":
    main()
