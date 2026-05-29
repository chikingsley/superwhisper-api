"""Inspect Superwhisper model network activity by running fixtures through the GUI.

Opens each fixture audio file in the Superwhisper macOS app, waits for the GUI
to write a recording row, then:

1. Checks whether the detected audio model and language model are already
   registered in the project's spec files.
2. If a text language model is missing, auto-probes the live API endpoint
   to discover its routing details and observed backend model.
3. Prints an actionable report showing what exists, what's missing, and
   exactly what code to add.
"""
from __future__ import annotations

import contextlib
import json
import plistlib
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from superwhisper_api.audio.models import AUDIO_MODELS
from superwhisper_api.auth import CACHE_DB, cached_auth
from superwhisper_api.text.client import (
    model_text,
    returned_model,
    sse_events,
)
from superwhisper_api.text.models import SUPERWHISPER_MODELS

SUPERWHISPER_DB = (
    Path.home() / "Library/Application Support/Superwhisper/database/superwhisper.sqlite"
)
SQLITE_READ_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class Transcript:
    """Represents a single Superwhisper GUI transcription result."""

    audio_path: Path
    recording_rowid: int
    recording_id: str
    datetime: str
    folder_name: str
    model_name: str
    mode_name: str
    duration: float | None
    processing_time: int | None
    transcript: str

    def as_dict(self) -> dict[str, object]:
        """Return the transcript as a plain dictionary."""
        return {
            "audio_path": str(self.audio_path),
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


def connect_superwhisper(db_path: Path) -> sqlite3.Connection:
    """Open a read-only SQLite connection to the Superwhisper database."""
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        timeout=SQLITE_READ_TIMEOUT_SECONDS,
        uri=True,
    )
    conn.row_factory = sqlite3.Row
    return conn


def latest_file_recording_rowid(db_path: Path) -> int:
    """Return the latest rowid from file recordings."""
    sql = "select coalesce(max(rowid), 0) from recording where fromFile = 1"
    with connect_superwhisper(db_path) as conn:
        return int(conn.execute(sql).fetchone()[0])


def file_rows_after(db_path: Path, after_rowid: int) -> list[sqlite3.Row]:
    """Return recording rows with transcripts after a given rowid."""
    sql = """
        select
            r.rowid as recordingRowid,
            hex(r.id) as recordingId,
            r.datetime,
            r.folderName,
            r.modelName,
            r.modeName,
            r.duration,
            r.processingTime,
            f.rawResult,
            f.result,
            f.llmResult
        from recording r
        join recording_fts f on f.recordingId = r.id
        where r.fromFile = 1
          and r.rowid > ?
        order by r.rowid asc
    """
    with connect_superwhisper(db_path) as conn:
        return list(conn.execute(sql, (after_rowid,)))


def transcript_from_row(audio_path: Path, row: sqlite3.Row) -> Transcript:
    """Build a Transcript from a database row."""
    transcript_text = row["rawResult"] or row["result"] or row["llmResult"] or ""
    return Transcript(
        audio_path=audio_path,
        recording_rowid=row["recordingRowid"],
        recording_id=row["recordingId"],
        datetime=row["datetime"],
        folder_name=row["folderName"],
        model_name=row["modelName"],
        mode_name=row["modeName"],
        duration=row["duration"],
        processing_time=row["processingTime"],
        transcript=transcript_text,
    )


def row_signature(row: sqlite3.Row) -> tuple[object, ...]:
    """Return a signature tuple for a single row."""
    return (
        row["recordingRowid"],
        row["duration"],
        row["processingTime"],
        row["rawResult"] or "",
        row["result"] or "",
        row["llmResult"] or "",
    )


def rows_signature(rows: list[sqlite3.Row]) -> tuple[tuple[object, ...], ...]:
    """Return signature tuples for multiple rows."""
    return tuple(row_signature(row) for row in rows)


def open_audio(app: str, audio_path: Path) -> None:
    """Open an audio file in a macOS application."""
    subprocess.run(["open", "-g", "-a", app, str(audio_path)], check=True)


def wait_for_file_transcripts(
    db_path: Path,
    after_rowid: int,
    audio_path: Path,
    timeout_seconds: int,
    poll_seconds: float,
    settle_seconds: float,
) -> list[Transcript]:
    """Poll the database until transcripts appear for an audio file."""
    deadline = time.monotonic() + timeout_seconds
    rows: list[sqlite3.Row] = []
    signature: tuple[tuple[object, ...], ...] = ()
    last_rowid = after_rowid
    last_change = time.monotonic()

    while time.monotonic() < deadline:
        current_rows = file_rows_after(db_path, after_rowid)
        if current_rows:
            current_last_rowid = current_rows[-1]["recordingRowid"]
            current_signature = rows_signature(current_rows)
            if (
                current_last_rowid != last_rowid
                or len(current_rows) != len(rows)
                or current_signature != signature
            ):
                rows = current_rows
                signature = current_signature
                last_rowid = current_last_rowid
                last_change = time.monotonic()
            elif time.monotonic() - last_change >= settle_seconds:
                return [transcript_from_row(audio_path, row) for row in rows]
        time.sleep(poll_seconds)

    if rows:
        return [transcript_from_row(audio_path, row) for row in rows]
    raise TimeoutError(f"Timed out waiting for Superwhisper rows: {audio_path}")


def latest_non_empty_transcript(rows: list[Transcript]) -> Transcript:
    """Return the most recent non-empty transcript."""
    for transcript in reversed(rows):
        if transcript.transcript.strip():
            return transcript
    return rows[-1]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = [
    _PROJECT_ROOT / "tests/fixtures/superwhisper/1779580688/output.wav",
    _PROJECT_ROOT / "tests/fixtures/superwhisper/1779497215/output.wav",
    _PROJECT_ROOT / "tests/fixtures/superwhisper/1779505994/output.wav",
    _PROJECT_ROOT / "tests/fixtures/superwhisper/1779570511/output.wav",
    _PROJECT_ROOT / "tests/fixtures/superwhisper/1779571988/output.wav",
]

_SUPERWHISPER_APP = "/Applications/superwhisper.app"
_TIMEOUT_SECONDS = 180
_POLL_SECONDS = 2.0
_SETTLE_SECONDS = 3.0


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_HEADER_INDICATORS = frozenset({
    "Authorization",
    "Content-Type",
    "Accept",
    "xi-api-key",
    "X-ID",
    "X-License",
    "X-Signature",
    "User-Agent",
})


def _looks_like_headers(d: dict[str, object]) -> bool:
    return bool({str(k) for k in d} & _HEADER_INDICATORS)


def _extract_header_dict(d: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, str):
            result[str(k)] = v
        elif isinstance(v, bytes):
            with contextlib.suppress(Exception):
                result[str(k)] = v.decode("utf-8")
    return result


def _extract_headers_from_plist(payload: object) -> dict[str, str] | None:
    """Recursively search a plist payload for a dict containing HTTP headers."""
    if isinstance(payload, dict):
        if _looks_like_headers(payload):
            return _extract_header_dict(payload) or None
        for value in payload.values():
            found = _extract_headers_from_plist(value)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _extract_headers_from_plist(value)
            if found is not None:
                return found
    return None


def _cache_entries_around(recording_time: str) -> list[dict[str, object]]:
    """Query CFURL cache for requests around the recording time."""
    if not CACHE_DB.exists():
        return [{"error": f"Cache DB not found: {CACHE_DB}"}]

    entries: list[dict[str, object]] = []
    with sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select
                r.request_key,
                r.time_stamp,
                b.request_object,
                d.receiver_data
            from cfurl_cache_response r
            join cfurl_cache_blob_data b on b.entry_id = r.entry_id
            left join cfurl_cache_receiver_data d on d.entry_id = r.entry_id
            where r.time_stamp >= datetime(?, '-10 minutes')
              and r.time_stamp <= datetime(?, '+10 minutes')
            order by r.time_stamp desc
            """,
            (recording_time, recording_time),
        ).fetchall()

        for row in rows:
            req_obj = row["request_object"]
            if isinstance(req_obj, str):
                req_obj = req_obj.encode()

            headers = None
            with contextlib.suppress(Exception):
                headers = _extract_headers_from_plist(plistlib.loads(req_obj))

            receiver_data = row["receiver_data"]
            preview = ""
            if isinstance(receiver_data, bytes):
                with contextlib.suppress(Exception):
                    preview = receiver_data.decode("utf-8", errors="replace")[:500]
                if not preview:
                    preview = repr(receiver_data)[:500]
            elif isinstance(receiver_data, str):
                preview = receiver_data[:500]

            entry: dict[str, object] = {
                "request_key": row["request_key"],
                "time_stamp": row["time_stamp"],
            }
            if headers:
                entry["headers"] = headers
            if preview:
                entry["receiver_data_preview"] = preview
            entries.append(entry)

    return entries


def _probe_text_model(
    model_key: str,
    endpoint_hint: str | None,
) -> dict[str, object]:
    """Probe a missing text model via the live API."""
    result: dict[str, object] = {
        "model_key": model_key,
        "probed": False,
        "status_code": None,
        "observed_model": None,
        "provider_guess": None,
        "path_guess": None,
        "text_snippet": None,
        "error": None,
    }

    try:
        auth = cached_auth()
    except RuntimeError as exc:
        result["error"] = f"auth failed: {exc}"
        return result

    # Prefer endpoint from cache; fall back to standard proxy
    path = endpoint_hint or "/v1/chat/completions"
    url = auth.base_url.rstrip("/") + path
    provider = _provider_from_path(path)

    payload = {
        "model": model_key,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Reply exactly ok"}],
    }

    try:
        resp = httpx.post(
            url,
            headers=auth.headers,
            json=payload,
            timeout=30,
        )
        result["status_code"] = resp.status_code
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        events = sse_events(resp.text)
        text = model_text(provider, events, resp.text) if events else resp.text[:200]
        observed = returned_model(provider, events) if events else None

        result["probed"] = True
        result["text_snippet"] = text[:200]
        result["observed_model"] = observed
        result["provider_guess"] = provider
        result["path_guess"] = path
    except Exception as exc:
        result["error"] = f"probe exception: {exc}"

    return result


def _provider_from_path(path: str) -> str:
    """Infer the text provider protocol from a Superwhisper route path."""
    if path.startswith("/anthropic/"):
        return "anthropic"
    if path.startswith("/gemini/"):
        return "gemini"
    return "openai"


def inspect_fixture(audio_path: Path) -> dict[str, object]:
    """Open one fixture through the GUI and inspect the resulting DB/cache trail."""
    _log(f"  → opening {audio_path.name}")

    before_rowid = latest_file_recording_rowid(SUPERWHISPER_DB)
    open_audio(_SUPERWHISPER_APP, audio_path)

    _log("  → waiting for Superwhisper to write recording row...")
    rows = wait_for_file_transcripts(
        SUPERWHISPER_DB,
        before_rowid,
        audio_path,
        _TIMEOUT_SECONDS,
        _POLL_SECONDS,
        _SETTLE_SECONDS,
    )

    transcript = latest_non_empty_transcript(rows)
    _log(
        f"  → got rowid {transcript.recording_rowid} "
        f"({transcript.model_name or 'unknown model'})",
    )

    with sqlite3.connect(f"file:{SUPERWHISPER_DB}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rec = conn.execute(
            """
            select
                r.rowid, r.datetime, r.modelKey, r.modelName,
                r.modeName, r.duration, r.processingTime,
                r.languageModelKey, r.languageModelName,
                f.rawResult, f.result, f.llmResult
            from recording r
            join recording_fts f on f.recordingId = r.id
            where r.rowid = ?
            """,
            (transcript.recording_rowid,),
        ).fetchone()

    if rec is None:
        return {
            "audio_path": str(audio_path),
            "error": "Recording row disappeared after polling.",
        }

    audio_key = rec["modelKey"]
    audio_name = rec["modelName"]
    lang_key = rec["languageModelKey"]
    lang_name = rec["languageModelName"]

    audio_known = audio_key in AUDIO_MODELS or any(
        spec.model_id == audio_key for spec in AUDIO_MODELS.values()
    )
    lang_known = bool(lang_key) and (
        lang_key in SUPERWHISPER_MODELS
        or any(spec.model_id == lang_key for spec in SUPERWHISPER_MODELS.values())
    )

    _log("  → scanning CFURL cache...")
    cache = _cache_entries_around(rec["datetime"])

    # Find the most likely text endpoint from cache
    endpoint_hint = None
    for entry in cache:
        rk = entry.get("request_key", "")
        if isinstance(rk, str) and ("/chat/completions" in rk or "/messages" in rk):
            endpoint_hint = rk.replace("https://api.superwhisper.com", "")
            break

    probe_result = None
    if lang_key and not lang_known:
        _log(f"  → language model '{lang_key}' not registered — auto-probing...")
        probe_result = _probe_text_model(lang_key, endpoint_hint)
        if probe_result.get("probed"):
            sc = probe_result["status_code"]
            om = probe_result["observed_model"]
            _log(f"  → probe OK: status={sc} model={om}")
        else:
            _log(f"  → probe FAILED: {probe_result.get('error')}")

    return {
        "audio_path": str(audio_path),
        "recording": {
            "rowid": rec["rowid"],
            "datetime": rec["datetime"],
            "model_key": audio_key,
            "model_name": audio_name,
            "mode_name": rec["modeName"],
            "duration": rec["duration"],
            "processing_time": rec["processingTime"],
            "language_model_key": lang_key,
            "language_model_name": lang_name,
            "transcript_preview": (
                rec["rawResult"] or rec["result"] or rec["llmResult"] or ""
            )[:200],
        },
        "known": {
            "audio_model": audio_known,
            "language_model": lang_known,
        },
        "cache_entries": cache,
        "probe": probe_result,
    }


def _suggest_code(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build a summary of missing models and suggested code."""
    audio_missing: set[str] = set()
    lang_missing: dict[str, dict[str, object]] = {}

    for r in results:
        known = r.get("known", {})
        rec = r.get("recording", {})
        probe = r.get("probe")

        if not known.get("audio_model"):
            audio_key = rec.get("model_key", "")
            if audio_key:
                audio_missing.add(audio_key)

        if not known.get("language_model"):
            lang_key = rec.get("language_model_key", "")
            if lang_key and probe and probe.get("probed"):
                lang_missing[lang_key] = probe
            elif lang_key:
                lang_missing[lang_key] = {"probed": False}

    suggestions: list[dict[str, object]] = []

    for key in sorted(audio_missing):
        suggestions.append({
            "type": "audio_model",
            "key": key,
            "action": "add to src/superwhisper_api/audio/models.py",
            "template": (
                f"{key.upper().replace('-', '_')} = AudioModelSpec(\n"
                f'    key="{key}",\n'
                f'    provider="???",\n'
                f'    model_id="{key}",\n'
                f")"
            ),
        })

    for key, probe in sorted(lang_missing.items()):
        if probe.get("probed"):
            suggestions.append({
                "type": "text_model",
                "key": key,
                "action": "add to src/superwhisper_api/text/models.py",
                "template": (
                    f"{key.upper().replace('-', '_').replace('.', '_')} = ModelSpec(\n"
                    f'    key="{key}",\n'
                    f'    model_id="{key}",\n'
                    f'    path="{probe.get("path_guess", "/v1/chat/completions")}",\n'
                    f'    provider="{probe.get("provider_guess", "openai")}",\n'
                    f'    observed_model="{probe.get("observed_model") or ""}",\n'
                    f")"
                ),
            })
        else:
            suggestions.append({
                "type": "text_model",
                "key": key,
                "action": (
                    "add to src/superwhisper_api/text/models.py "
                    "(probe failed — manual inspection required)"
                ),
                "template": (
                    f"{key.upper().replace('-', '_').replace('.', '_')} = ModelSpec(\n"
                    f'    key="{key}",\n'
                    f'    model_id="{key}",\n'
                    f'    path="/v1/chat/completions",  # <-- verify\n'
                    f'    provider="openai",  # <-- verify\n'
                    f")"
                ),
            })

    return suggestions


def main() -> int:
    """Run all fixture files through Superwhisper and inspect model activity."""
    _log("=" * 60)
    _log("Superwhisper new-model-inspect")
    _log("=" * 60)

    results: list[dict[str, object]] = []
    for idx, fixture in enumerate(FIXTURES, 1):
        if not fixture.exists():
            _log(f"\nSKIP {fixture}: not found")
            results.append({"audio_path": str(fixture), "error": "fixture not found"})
            continue

        _log(f"\n[{idx}/{len(FIXTURES)}] {fixture.name}")
        try:
            result = inspect_fixture(fixture)
            results.append(result)
        except Exception as exc:
            _log(f"  ✗ error: {exc}")
            results.append({"audio_path": str(fixture), "error": str(exc)})

    _log("\n" + "=" * 60)
    _log("BUILDING ACTIONABLE REPORT")
    _log("=" * 60)

    suggestions = _suggest_code(results)

    report = {
        "results": results,
        "suggestions": suggestions,
        "summary": {
            "total_fixtures": len(FIXTURES),
            "successful_inspections": sum(1 for r in results if "error" not in r),
            "failed_inspections": sum(1 for r in results if "error" in r),
            "missing_audio_models": sorted(
                {s["key"] for s in suggestions if s["type"] == "audio_model"}
            ),
            "missing_text_models": sorted(
                {s["key"] for s in suggestions if s["type"] == "text_model"}
            ),
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
