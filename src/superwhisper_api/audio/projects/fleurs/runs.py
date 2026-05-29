"""Resumable transcription runs and WER/CER scoring over SQLite samples."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jiwer

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.projects.fleurs.db import connect, ensure_schema, utc_now
from superwhisper_api.audio.projects.fleurs.models import FleursProject, RunStats
from superwhisper_api.audio.transcribe import create_process_fn
from superwhisper_api.batch import bounded_map

if TYPE_CHECKING:
    from superwhisper_api.audio.projects.fleurs.models import NormalizeFn
    from superwhisper_api.audio.transcribe import ProcessFn, TranscriptResult

DEFAULT_HIGH_WER_THRESHOLD = 0.35
DEFAULT_HIGH_CER_THRESHOLD = 0.10


def ensure_run(
    conn: sqlite3.Connection,
    *,
    run_name: str,
    dataset: str,
    config: str,
    split: str,
    model: str,
    language: str | None,
) -> int:
    """Create or return a resumable transcription run."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT run_id FROM transcription_runs WHERE run_name = ?",
        (run_name,),
    ).fetchone()
    if row:
        return int(row[0])
    now = utc_now()
    with conn:
        cursor = conn.execute(
            """
            INSERT INTO transcription_runs (
                run_name, dataset, config, split, model_key, language, status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (run_name, dataset, config, split, model, language, now, now),
        )
    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return a run_id for the inserted run")
    return int(cursor.lastrowid)


def run_id_for_name(conn: sqlite3.Connection, run_name: str) -> int:
    """Return the run id for an existing run name."""
    ensure_schema(conn)
    row = conn.execute(
        "SELECT run_id FROM transcription_runs WHERE run_name = ?",
        (run_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown transcription run {run_name!r}")
    return int(row[0])


def pending_samples(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    dataset: str,
    config: str,
    split: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Return samples not yet represented in this transcription run."""
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    rows = conn.execute(
        f"""
        SELECT sample.sample_id, sample.audio_path, sample.normalized_ref_text
        FROM fleurs_samples sample
        LEFT JOIN transcription_results result
          ON result.sample_id = sample.sample_id AND result.run_id = ?
        WHERE sample.dataset = ? AND sample.config = ? AND sample.split = ?
          AND TRIM(COALESCE(sample.normalized_ref_text, '')) <> ''
          AND result.result_id IS NULL
        ORDER BY sample.sample_id
        {limit_sql}
        """,
        (run_id, dataset, config, split),
    ).fetchall()
    return [
        {
            "sample_id": int(sample_id),
            "audio_path": str(audio_path),
            "normalized_ref_text": str(normalized_ref_text or ""),
        }
        for sample_id, audio_path, normalized_ref_text in rows
    ]


def score(
    normalized_ref_text: str,
    normalized_pred_text: str,
) -> tuple[float | None, float | None]:
    """Return WER/CER over normalized text."""
    if not normalized_ref_text:
        return None, None
    return float(jiwer.wer(normalized_ref_text, normalized_pred_text)), float(
        jiwer.cer(normalized_ref_text, normalized_pred_text)
    )


def error_profile(
    *,
    normalized_ref_text: str,
    normalized_pred_text: str,
    wer: float | None,
    cer: float | None,
    high_wer_threshold: float = DEFAULT_HIGH_WER_THRESHOLD,
    high_cer_threshold: float = DEFAULT_HIGH_CER_THRESHOLD,
) -> str:
    """Classify an objective WER/CER relationship."""
    if not normalized_ref_text:
        return "empty_ref"
    if not normalized_pred_text:
        return "empty_pred"
    if normalized_ref_text == normalized_pred_text:
        return "exact"
    if wer is None or cer is None:
        return "unscored"
    high_wer = wer > high_wer_threshold
    high_cer = cer > high_cer_threshold
    if high_wer and high_cer:
        return "high_wer_high_cer"
    if high_wer:
        return "high_wer_low_cer"
    if high_cer:
        return "low_wer_high_cer"
    return "low_wer_low_cer"


def record_result(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    sample: dict[str, Any],
    result: TranscriptResult,
    normalize: NormalizeFn,
) -> None:
    """Persist one successful or failed transcription result."""
    now = utc_now()
    payload = result.as_dict()
    error = str(payload.get("error") or "")
    pred_text = str(payload.get("transcript") or "")
    normalized_pred_text = "" if error else (normalize(pred_text) or "")
    normalized_ref_text = str(sample["normalized_ref_text"])
    wer, cer = (
        (None, None)
        if error
        else score(normalized_ref_text, normalized_pred_text)
    )
    profile = (
        ""
        if error
        else error_profile(
            normalized_ref_text=normalized_ref_text,
            normalized_pred_text=normalized_pred_text,
            wer=wer,
            cer=cer,
        )
    )
    with conn:
        conn.execute(
            """
            INSERT INTO transcription_results (
                run_id, sample_id, provider, model_key, model_id, pred_text,
                normalized_pred_text, wer, cer, error_profile, recording_id, duration,
                processing_time,
                raw_response_json, error, attempts, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(run_id, sample_id) DO UPDATE SET
                provider = excluded.provider,
                model_key = excluded.model_key,
                model_id = excluded.model_id,
                pred_text = excluded.pred_text,
                normalized_pred_text = excluded.normalized_pred_text,
                wer = excluded.wer,
                cer = excluded.cer,
                error_profile = excluded.error_profile,
                recording_id = excluded.recording_id,
                duration = excluded.duration,
                processing_time = excluded.processing_time,
                raw_response_json = excluded.raw_response_json,
                error = excluded.error,
                attempts = transcription_results.attempts + 1,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                sample["sample_id"],
                payload.get("provider"),
                payload.get("model_key"),
                payload.get("model_id"),
                pred_text,
                normalized_pred_text,
                wer,
                cer,
                profile,
                payload.get("recording_id"),
                payload.get("duration"),
                payload.get("processing_time"),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                error or None,
                now,
                now,
            ),
        )


def run_transcriptions(
    project: FleursProject,
    *,
    database: Path,
    dataset: str,
    config: str,
    split: str,
    run_name: str,
    model: str,
    language: str | None,
    key: str | None,
    limit: int,
    max_workers: int,
    process: ProcessFn | None = None,
) -> RunStats:
    """Run pending SQLite samples through Scribe/provider transcription."""
    conn = connect(database)
    ok = 0
    failed = 0
    submitted = 0
    try:
        run_id = ensure_run(
            conn,
            run_name=run_name,
            dataset=dataset,
            config=config,
            split=split,
            model=model,
            language=language,
        )
        samples = pending_samples(
            conn,
            run_id=run_id,
            dataset=dataset,
            config=config,
            split=split,
            limit=limit,
        )
        resolved = (
            process
            if process is not None
            else create_process_fn(audio_model(model), key, language=language)
        )
        for sample, result in bounded_map(
            samples,
            lambda s: resolved(Path(s["audio_path"])),
            max_workers=max_workers,
        ):
            submitted += 1
            record_result(
                conn,
                run_id=run_id,
                sample=sample,
                result=result,
                normalize=project.normalize,
            )
            if getattr(result, "error", ""):
                failed += 1
                print(
                    f"FAIL {sample['sample_id']}: {getattr(result, 'error', '')}",
                    file=sys.stderr,
                )
            else:
                ok += 1
                text = str(getattr(result, "transcript", ""))[:80].replace("\n", " ")
                print(f"OK {sample['sample_id']}: {text}", file=sys.stderr)
        pending = len(
            pending_samples(
                conn,
                run_id=run_id,
                dataset=dataset,
                config=config,
                split=split,
                limit=0,
            )
        )
        return RunStats(
            database=str(database),
            run_id=run_id,
            run_name=run_name,
            ok=ok,
            failed=failed,
            submitted=submitted,
            pending=pending,
        )
    finally:
        conn.close()
