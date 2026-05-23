from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from super_whisperer_cli.superwhisper import (
    SUPERWHISPER_DB,
    latest_file_recording_rowid,
    latest_non_empty_transcript,
    open_audio,
    wait_for_file_transcripts,
)

SUPERWHISPER_APP = "/Applications/superwhisper.app"
TIMEOUT_SECONDS = 180
TIMEOUT_RETRIES = 3
OPEN_RETRIES = 3
EMPTY_RETRIES = 3
POLL_SECONDS = 2.0
SETTLE_SECONDS = 3.0
LogFn = Callable[[str], None]
OpenAudioFn = Callable[[Path], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="super-whisperer",
        description="Run a Superwhisper manifest job and write transcript rows.",
    )
    parser.add_argument("--paths-file", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--fail-jsonl", type=Path, required=True)
    return parser


def last_written_audio_path(path: Path) -> str | None:
    path = path.expanduser()
    if not path.exists():
        return None

    last_audio_path = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            audio_path = payload.get("audio_path")
            if isinstance(audio_path, str):
                last_audio_path = audio_path
    return last_audio_path


def load_skip_paths(paths_files: Sequence[Path]) -> set[str]:
    skipped: set[str] = set()
    for paths_file in paths_files:
        path = paths_file.expanduser()
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    skipped.add(stripped)
    return skipped


def iter_pending_audio_paths(
    paths_file: Path,
    jsonl: Path,
    log: LogFn = print,
    skip_paths_files: Sequence[Path] = (),
) -> Iterable[Path]:
    paths_file = paths_file.expanduser()
    last_audio_path = last_written_audio_path(jsonl)
    skip_paths = load_skip_paths(skip_paths_files)
    skipped = 0
    skipped_reserved = 0
    found_resume_point = last_audio_path is None

    with paths_file.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not found_resume_point:
                skipped += 1
                if stripped == last_audio_path:
                    found_resume_point = True
                    log(f"skipping {skipped} already-written paths from {jsonl}")
                continue
            if stripped in skip_paths:
                skipped_reserved += 1
                continue
            else:
                yield Path(stripped)

    if last_audio_path is not None and not found_resume_point:
        log(f"resume path not found in {paths_file}; no new paths yielded")
    if skipped_reserved:
        log(f"skipped {skipped_reserved} reserved paths from {paths_file}")


def write_failure(path: Path, audio_path: Path, error: str, attempts: int) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_path": str(audio_path),
        "error": error,
        "attempts": attempts,
        "created_at": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, transcript) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(transcript.as_dict(), ensure_ascii=False) + "\n")


def run_job(
    paths_file: Path,
    jsonl: Path,
    fail_jsonl: Path,
    db_path: Path = SUPERWHISPER_DB,
    open_audio_fn: OpenAudioFn | None = None,
    log: LogFn | None = None,
    skip_paths_files: Sequence[Path] = (),
) -> int:
    if log is None:
        def log(message: str) -> None:
            print(message, file=sys.stderr)

    if open_audio_fn is None:
        def open_audio_fn(audio_path: Path) -> None:
            open_audio(SUPERWHISPER_APP, audio_path)

    if not paths_file.expanduser().exists():
        log(f"No audio paths found in {paths_file}.")
        return 1

    after_rowid = latest_file_recording_rowid(db_path)
    log(f"starting after Superwhisper file recording rowid: {after_rowid}")

    processed = 0
    for audio_path in iter_pending_audio_paths(paths_file, jsonl, log, skip_paths_files):
        processed += 1
        transcript = None
        max_attempts = max(EMPTY_RETRIES, TIMEOUT_RETRIES, OPEN_RETRIES) + 1
        last_error = ""
        for attempt in range(max_attempts):
            try:
                open_audio_fn(audio_path)
            except subprocess.CalledProcessError as exc:
                last_error = f"open failed: {exc}"
                after_rowid = latest_file_recording_rowid(db_path)
                if attempt < OPEN_RETRIES:
                    log(f"open attempt {attempt + 1}; retrying {audio_path}: {exc}")
                    time.sleep(POLL_SECONDS)
                    continue
                raise RuntimeError(
                    f"{last_error}; stopping because file-open failures mean the "
                    "Superwhisper app or LaunchServices is unhealthy"
                ) from exc

            try:
                rows = wait_for_file_transcripts(
                    db_path,
                    after_rowid,
                    audio_path,
                    TIMEOUT_SECONDS,
                    POLL_SECONDS,
                    SETTLE_SECONDS,
                )
            except TimeoutError as exc:
                last_error = str(exc)
                after_rowid = latest_file_recording_rowid(db_path)
                if attempt < TIMEOUT_RETRIES:
                    log(f"timeout attempt {attempt + 1}; retrying {audio_path}: {exc}")
                    continue
                write_failure(fail_jsonl, audio_path, last_error, attempt + 1)
                break

            after_rowid = rows[-1].recording_rowid
            transcript = latest_non_empty_transcript(rows)
            if transcript.transcript.strip() or attempt >= EMPTY_RETRIES:
                break
            log(f"empty transcript attempt {attempt + 1}; retrying {audio_path}")

        if transcript is None:
            continue

        write_jsonl(jsonl, transcript)

    if processed == 0:
        log(f"No pending audio paths found in {paths_file}.")

    return 0


def main() -> int:
    args = build_parser().parse_args()
    return run_job(args.paths_file, args.jsonl, args.fail_jsonl)
