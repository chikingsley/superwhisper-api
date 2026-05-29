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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TYPE_CHECKING

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import (
    ProcessFn,
    TranscriptResult,
    create_process_fn,
    warn_if_key_ignored,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future


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


def _submit_next(
    pool: ThreadPoolExecutor,
    pending_paths: Iterator[Path],
    futures: set[Future[TranscriptResult]],
    process: ProcessFn,
    submitted_ref: list[int],
) -> bool:
    """Pull the next audio path and submit it to the thread pool."""
    try:
        audio = next(pending_paths)
    except StopIteration:
        return False
    futures.add(pool.submit(process, audio))
    submitted_ref[0] += 1
    return True


def _handle_done(
    done: set[Future[TranscriptResult]],
    jsonl: Path,
    fail_jsonl: Path | None,
    counters: list[int],
) -> None:
    """Process completed futures and write results.  counters = [ok, fail]."""
    for fut in done:
        t = fut.result()
        error = getattr(t, "error", "")
        if error:
            counters[1] += 1
            if fail_jsonl:
                _write_jsonl(fail_jsonl, t)
            print(f"FAIL {t.audio_path}: {error}", file=sys.stderr)
        else:
            counters[0] += 1
            _write_jsonl(jsonl, t)
            transcript_text = getattr(t, "transcript", "")
            snippet = transcript_text[:80].replace("\n", " ")
            print(f"OK {t.audio_path}: {snippet}", file=sys.stderr)


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

    pending = _iter_pending(paths_file, jsonl)
    submitted = [0]
    counters = [0, 0]  # [ok, fail]
    max_in_flight = max(max_workers * 4, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: set[Future[TranscriptResult]] = set()
        while len(futures) < max_in_flight and _submit_next(
            pool, pending, futures, process, submitted
        ):
            pass
        if not futures:
            print("No pending audio files.", file=sys.stderr)
            return 0

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            _handle_done(done, jsonl, fail_jsonl, counters)
            while len(futures) < max_in_flight and _submit_next(
                pool, pending, futures, process, submitted
            ):
                pass

    print(
        f"Done: {counters[0]} ok, {counters[1]} failed, {submitted[0]} submitted",
        file=sys.stderr,
    )
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
