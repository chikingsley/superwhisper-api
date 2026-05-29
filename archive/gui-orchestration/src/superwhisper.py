from __future__ import annotations

import sqlite3
import subprocess
import time
from pathlib import Path

from superwhisper_api.models import Transcript

SUPERWHISPER_DB = (
    Path.home() / "Library/Application Support/superwhisper/database/superwhisper.sqlite"
)
SQLITE_READ_TIMEOUT_SECONDS = 30.0


def connect_superwhisper(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        timeout=SQLITE_READ_TIMEOUT_SECONDS,
        uri=True,
    )
    conn.row_factory = sqlite3.Row
    return conn


def latest_file_recording_rowid(db_path: Path) -> int:
    sql = "select coalesce(max(rowid), 0) from recording where fromFile = 1"
    with connect_superwhisper(db_path) as conn:
        return int(conn.execute(sql).fetchone()[0])


def file_rows_after(db_path: Path, after_rowid: int) -> list[sqlite3.Row]:
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
    return (
        row["recordingRowid"],
        row["duration"],
        row["processingTime"],
        row["rawResult"] or "",
        row["result"] or "",
        row["llmResult"] or "",
    )


def rows_signature(rows: list[sqlite3.Row]) -> tuple[tuple[object, ...], ...]:
    return tuple(row_signature(row) for row in rows)


def open_audio(app: str, audio_path: Path) -> None:
    subprocess.run(["open", "-g", "-a", app, str(audio_path)], check=True)


def wait_for_file_transcripts(
    db_path: Path,
    after_rowid: int,
    audio_path: Path,
    timeout_seconds: int,
    poll_seconds: float,
    settle_seconds: float,
) -> list[Transcript]:
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
    for transcript in reversed(rows):
        if transcript.transcript.strip():
            return transcript
    return rows[-1]
