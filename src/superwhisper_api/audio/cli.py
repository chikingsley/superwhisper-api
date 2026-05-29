"""CLI for transcribing audio through multiple provider backends.

Supported providers:
  - elevenlabs  (direct API key extracted from Superwhisper cache)
  - deepgram    (Superwhisper proxy with signed headers)
  - s1          (S1 Voice via v2 inference key)
  - ultra       (Ultra via v1 inference key)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import (
    ProcessFn,
    TranscriptResult,
    create_process_fn,
    warn_if_key_ignored,
)
from superwhisper_api.batch import bounded_map

if TYPE_CHECKING:
    from collections.abc import Iterator


def _written_audio_paths(path: Path) -> set[str]:
    """Collect audio_path values already written to a JSONL file."""
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


def _iter_pending(paths_file: Path, jsonl: Path) -> Iterator[Path]:
    """Yield audio paths that have not yet been written to jsonl."""
    completed = _written_audio_paths(jsonl)
    skipped = 0

    def _pending_paths() -> Iterator[Path]:
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

    return _pending_paths()


def _write_jsonl(path: Path, t: TranscriptResult) -> None:
    """Append a transcript/failure record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(t.as_dict(), ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for superwhisper-audio."""
    p = argparse.ArgumentParser(
        description="Transcribe audio via Superwhisper-backed providers (Single or Batch Mode)"
    )
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
    p.add_argument(
        "--key",
        help="ElevenLabs API key (auto-extracted from Superwhisper if omitted)",
    )
    p.add_argument("--model", default="scribe-v2", help="Model key (default: scribe-v2)")
    p.add_argument("--language", help="Language code hint (e.g. fas, eng)")
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers in batch mode (default: 4)",
    )
    return p


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate CLI arguments and exit with a message if invalid."""
    if not args.audio and not args.paths_file:
        parser.error("Either a single 'audio' file or '--paths-file' must be specified.")
    if args.audio and args.paths_file:
        parser.error("Cannot specify both a single 'audio' file and '--paths-file'.")
    if args.paths_file and not args.jsonl:
        parser.error("--jsonl must be specified when using --paths-file.")


def _run_single(
    audio: Path, process: ProcessFn, *, pretty: bool = False
) -> int:
    """Transcribe a single audio file and print the result."""
    if not audio.exists():
        print(f"File not found: {audio}", file=sys.stderr)
        return 1

    try:
        result = process(audio)
        indent = 2 if pretty else None
        print(json.dumps(result.as_dict(), indent=indent, ensure_ascii=False))
        if getattr(result, "error", ""):
            return 1
    except Exception as exc:
        print(f"Error transcribing file: {exc}", file=sys.stderr)
        return 1

    return 0


def _run_batch(
    paths_file: Path,
    jsonl: Path,
    fail_jsonl: Path | None,
    max_workers: int,
    process: ProcessFn,
) -> int:
    """Transcribe a batch of audio files and write results to JSONL."""
    if not paths_file.exists():
        print(f"Paths file not found: {paths_file}", file=sys.stderr)
        return 1

    ok = 0
    failed = 0
    for _audio, t in bounded_map(
        _iter_pending(paths_file, jsonl), process, max_workers=max_workers
    ):
        error = getattr(t, "error", "")
        if error:
            failed += 1
            if fail_jsonl:
                _write_jsonl(fail_jsonl, t)
            print(f"FAIL {t.audio_path}: {error}", file=sys.stderr)
        else:
            ok += 1
            _write_jsonl(jsonl, t)
            snippet = getattr(t, "transcript", "")[:80].replace("\n", " ")
            print(f"OK {t.audio_path}: {snippet}", file=sys.stderr)

    if ok == 0 and failed == 0:
        print("No pending audio files.", file=sys.stderr)
        return 0
    print(f"Done: {ok} ok, {failed} failed, {ok + failed} submitted", file=sys.stderr)
    return 0


def main() -> int:
    """Entry point for the superwhisper-audio CLI."""
    p = _build_parser()
    args = p.parse_args()
    _validate_args(args, p)

    try:
        spec = audio_model(args.model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    warn_if_key_ignored(spec.provider, args.key)
    try:
        process = create_process_fn(spec, args.key, language=args.language)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.audio:
        return _run_single(args.audio, process, pretty=args.pretty)

    return _run_batch(
        args.paths_file,
        args.jsonl,
        args.fail_jsonl,
        args.max_workers,
        process,
    )


if __name__ == "__main__":
    raise SystemExit(main())
