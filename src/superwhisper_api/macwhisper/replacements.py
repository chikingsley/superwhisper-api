"""Agent-driven helper for growing MacWhisper's Global Replace dictionary.

This tool is deliberately "dumb": the CLI agent (Claude Code, Codex, etc.) is the
intelligence. ``learn`` transcribes recent (or given) audio with a fast local
model and the accurate cloud model and prints both transcripts plus the current
Global Replace list. The agent reads that, proposes ``original -> replacement``
pairs in chat, and on your approval calls ``apply`` with them.

Model and path defaults are module constants below — edit them here rather than
passing flags.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import create_process_fn

# Defaults — change here, not via flags.
PARAKEET_MODEL = "parakeet-pro:nvidia_parakeet-v3_494MB"
SCRIBE_MODEL = "scribe-v2"
MACWHISPER_DB = Path.home() / "Library/Application Support/MacWhisper/Database/main.sqlite"
MACWHISPER_MEDIA_DIR = Path.home() / "Library/Application Support/MacWhisper/Database/ExternalMedia"
MACWHISPER_DEFAULTS_DOMAIN = "com.goodsnooze.MacWhisper"


# --- MacWhisper history (read-only) ---------------------------------------


def _latest_audio_paths(limit: int) -> list[Path]:
    """Return media paths for the most recent non-deleted recordings."""
    query = """
        SELECT m.filename
        FROM dictation d
        LEFT JOIN mediafile m ON m.dictationID = d.id
        WHERE d.dateDeleted IS NULL AND m.filename IS NOT NULL
        ORDER BY d.dateCreated DESC
        LIMIT ?
    """
    with sqlite3.connect(f"file:{MACWHISPER_DB}?mode=ro", uri=True) as conn:
        rows = conn.execute(query, (limit,)).fetchall()
    return [MACWHISPER_MEDIA_DIR / str(row[0]) for row in rows]


# --- Transcription --------------------------------------------------------


def _transcribe_with_mw(audio: Path) -> str:
    """Transcribe with the local MacWhisper ``mw`` CLI."""
    result = subprocess.run(
        ["mw", "transcribe", "--model", PARAKEET_MODEL, str(audio)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _transcribe_with_scribe(audio: Path) -> str:
    """Transcribe via the cloud Scribe model."""
    process = create_process_fn(audio_model(SCRIBE_MODEL), key=None)
    result = process(audio)
    error = getattr(result, "error", "")
    if error:
        raise RuntimeError(f"Scribe transcription failed for {audio}: {error}")
    return str(getattr(result, "transcript", "")).strip()


# --- Global Replace store (defaults plist) --------------------------------


def _read_global_replace_items() -> list[dict[str, str]]:
    """Read MacWhisper's Global Replace list (returns [] if none is set)."""
    try:
        exported = subprocess.run(
            ["defaults", "export", MACWHISPER_DEFAULTS_DOMAIN, "-"],
            check=True,
            capture_output=True,
        )
        extracted = subprocess.run(
            ["plutil", "-extract", "globalReplaceList", "json", "-o", "-", "-"],
            input=exported.stdout,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [json.loads(item) for item in json.loads(extracted.stdout)]


def _write_global_replace_items(items: list[dict[str, str]]) -> None:
    """Write the Global Replace list back into MacWhisper's preferences."""
    encoded = [json.dumps(item, separators=(",", ":"), ensure_ascii=False) for item in items]
    subprocess.run(
        [
            "defaults",
            "write",
            MACWHISPER_DEFAULTS_DOMAIN,
            "globalReplaceList",
            "-array",
            *encoded,
        ],
        check=True,
    )


def _merge_items(
    existing: list[dict[str, str]],
    pairs: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int, int]:
    """Merge pairs into existing, updating an existing original's replacement.

    Returns (merged, added, updated). Matching is case-insensitive on original.
    """
    by_original = {item.get("original", "").casefold(): item for item in existing}
    merged = list(existing)
    added = 0
    updated = 0
    for pair in pairs:
        key = pair["original"].casefold()
        current = by_original.get(key)
        if current is not None:
            if current.get("replacement") != pair["replacement"]:
                current["replacement"] = pair["replacement"]
                updated += 1
            continue
        item = {
            "id": str(uuid.uuid4()).upper(),
            "original": pair["original"],
            "replacement": pair["replacement"],
        }
        merged.append(item)
        by_original[key] = item
        added += 1
    return merged, added, updated


def _remove_items(
    existing: list[dict[str, str]],
    originals: list[str],
) -> tuple[list[dict[str, str]], int]:
    """Drop items whose original matches (case-insensitive). Returns (merged, removed)."""
    remove_keys = {original.casefold() for original in originals}
    merged = [
        item for item in existing if item.get("original", "").casefold() not in remove_keys
    ]
    return merged, len(existing) - len(merged)


def _parse_pairs(raw: str) -> list[dict[str, str]]:
    """Parse a JSON array of {original, replacement} objects."""
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of {original, replacement} objects")
    pairs: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"each entry must be an object, got {entry!r}")
        original = entry.get("original")
        replacement = entry.get("replacement")
        if not isinstance(original, str) or not isinstance(replacement, str):
            raise ValueError(f"entry needs string 'original' and 'replacement': {entry!r}")
        pairs.append({"original": original, "replacement": replacement})
    return pairs


def _parse_originals(raw: str) -> list[str]:
    """Parse a JSON array of original strings to remove."""
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of original strings")
    originals: list[str] = []
    for entry in data:
        if not isinstance(entry, str):
            raise ValueError(f"each entry must be a string, got {entry!r}")
        originals.append(entry)
    return originals


# --- Commands -------------------------------------------------------------


def _cmd_learn(args: argparse.Namespace) -> int:
    """Transcribe recent/given audio with both models and print context as JSON."""
    audio_paths = (
        [path.expanduser() for path in args.audio]
        if args.audio
        else _latest_audio_paths(args.latest)
    )
    if not audio_paths:
        print("No audio to transcribe.", file=sys.stderr)
        return 1

    recordings: list[dict[str, Any]] = []
    for audio in audio_paths:
        print(f"Transcribing {audio.name}", file=sys.stderr)
        recordings.append(
            {
                "audio_path": str(audio),
                "parakeet": _transcribe_with_mw(audio),
                "scribe": _transcribe_with_scribe(audio),
            }
        )

    payload = {
        "recordings": recordings,
        "current_replacements": _read_global_replace_items(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    """Apply approved replacement pairs (JSON array) into MacWhisper."""
    try:
        pairs = _parse_pairs(args.pairs)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not pairs:
        print("Nothing to apply.", file=sys.stderr)
        return 0

    existing = _read_global_replace_items()
    merged, added, updated = _merge_items(existing, pairs)
    _write_global_replace_items(merged)
    print(f"Added {added} replacement(s) and updated {updated}.")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    """Remove replacements by original (JSON array of strings) from MacWhisper."""
    try:
        originals = _parse_originals(args.originals)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not originals:
        print("Nothing to remove.", file=sys.stderr)
        return 0

    existing = _read_global_replace_items()
    merged, removed = _remove_items(existing, originals)
    _write_global_replace_items(merged)
    print(f"Removed {removed} replacement(s).")
    return 0


def add_learn_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``learn`` subcommand."""
    parser = subparsers.add_parser(
        "learn",
        help="Transcribe recent/given audio with both models; print transcripts "
        "and current replacements for an agent to review.",
    )
    parser.add_argument("audio", nargs="*", type=Path, help="Audio files (default: latest).")
    parser.add_argument("--latest", type=int, default=1, help="How many recent recordings to use.")
    parser.set_defaults(func=_cmd_learn)
    return parser


def add_apply_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``apply`` subcommand."""
    parser = subparsers.add_parser(
        "apply",
        help='Apply a JSON array of {"original","replacement"} pairs into MacWhisper.',
    )
    parser.add_argument(
        "pairs",
        help='JSON array, e.g. \'[{"original":"Deep Gram","replacement":"Deepgram"}]\'',
    )
    parser.set_defaults(func=_cmd_apply)
    return parser


def add_remove_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register the ``remove`` subcommand."""
    parser = subparsers.add_parser(
        "remove",
        help="Remove replacements by original (JSON array of strings) from MacWhisper.",
    )
    parser.add_argument(
        "originals",
        help='JSON array of originals, e.g. \'["Deep Gram","Chi Bazor"]\'',
    )
    parser.set_defaults(func=_cmd_remove)
    return parser
