"""SQLite workflow for streaming Google FLEURS samples and transcribing them."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tarfile
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jiwer
from huggingface_hub import hf_hub_download

from superwhisper_api.audio.models import audio_model
from superwhisper_api.audio.transcribe import TranscriptResult, create_process_fn
from superwhisper_api.text.client import SuperwhisperClient

if TYPE_CHECKING:
    from superwhisper_api.audio.transcribe import ProcessFn

NormalizeFn = Callable[[str], str | None]
AuditFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]
DEFAULT_FLEURS_SOURCE_DIR = Path("/Volumes/simons-enjoyment/Hugging Face/google-fleurs")
DEFAULT_HIGH_WER_THRESHOLD = 0.35
DEFAULT_HIGH_CER_THRESHOLD = 0.10
DEFAULT_AUDIT_MODEL = "gpt-5.4-mini"

SCRIBE_AUDIT_CATEGORIES = {
    "substitution",
    "deletion",
    "insertion",
    "mixed_edit",
    "content_mismatch",
    "unclear",
}

SCRIBE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": sorted(SCRIBE_AUDIT_CATEGORIES)},
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
    "additionalProperties": False,
}

SCRIBE_AUDIT_PROMPT = """Classify a Google FLEURS ASR reference/prediction difference.

Return JSON with exactly:
category, reason

category must be one of:
- substitution: reference and prediction mostly align, but words are replaced.
- deletion: prediction omits words from the reference.
- insertion: prediction adds words not in the reference.
- mixed_edit: more than one edit type is materially present.
- content_mismatch: text appears substantially different, beyond a local edit.
- unclear: text alone is insufficient.

Use the provided normalized text and edit counts. Do not add markdown.

Input:
{row_json}
"""


@dataclass(frozen=True)
class FleursProject:
    """Defaults for one Google FLEURS language project."""

    name: str
    dataset: str
    config: str
    split: str
    database: Path
    media_dir: Path
    language: str
    normalize: NormalizeFn


@dataclass(frozen=True)
class IngestStats:
    """Summary for streamed sample ingestion."""

    database: str
    dataset: str
    config: str
    split: str
    added: int
    skipped_existing: int
    scanned: int


@dataclass(frozen=True)
class RunStats:
    """Summary for a transcription run."""

    database: str
    run_id: int
    run_name: str
    ok: int
    failed: int
    submitted: int
    pending: int


@dataclass(frozen=True)
class AuditStats:
    """Summary for Scribe audit classification."""

    database: str
    run_id: int
    run_name: str
    model: str
    deterministic: int
    api: int
    failed: int
    pending: int


def utc_now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(UTC).isoformat()


def connect(database: Path) -> sqlite3.Connection:
    """Open the project database with production-friendly pragmas."""
    database = database.expanduser()
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the Google FLEURS sample/run/result tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS fleurs_samples (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            config TEXT NOT NULL,
            split TEXT NOT NULL,
            hf_id TEXT NOT NULL,
            audio_path TEXT NOT NULL,
            sample_rate INTEGER NOT NULL,
            num_samples INTEGER NOT NULL,
            ref_text TEXT NOT NULL,
            normalized_ref_text TEXT,
            raw_row_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(dataset, config, split, hf_id)
        );

        CREATE TABLE IF NOT EXISTS transcription_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL UNIQUE,
            dataset TEXT NOT NULL,
            config TEXT NOT NULL,
            split TEXT NOT NULL,
            model_key TEXT NOT NULL,
            language TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS transcription_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            sample_id INTEGER NOT NULL,
            provider TEXT,
            model_key TEXT,
            model_id TEXT,
            pred_text TEXT,
            normalized_pred_text TEXT,
            wer REAL,
            cer REAL,
            error_profile TEXT,
            recording_id TEXT,
            duration REAL,
            processing_time INTEGER,
            raw_response_json TEXT NOT NULL,
            error TEXT,
            attempts INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES transcription_runs(run_id),
            FOREIGN KEY(sample_id) REFERENCES fleurs_samples(sample_id),
            UNIQUE(run_id, sample_id)
        );

        CREATE INDEX IF NOT EXISTS idx_fleurs_samples_project
            ON fleurs_samples(dataset, config, split);
        CREATE INDEX IF NOT EXISTS idx_transcription_results_run
            ON transcription_results(run_id, sample_id);

        CREATE TABLE IF NOT EXISTS scribe_audits (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            sample_id INTEGER NOT NULL,
            audit_model TEXT NOT NULL,
            category TEXT NOT NULL,
            reason TEXT NOT NULL,
            source TEXT NOT NULL,
            raw_audit_json TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES transcription_runs(run_id),
            FOREIGN KEY(sample_id) REFERENCES fleurs_samples(sample_id),
            UNIQUE(run_id, sample_id, audit_model)
        );

        CREATE INDEX IF NOT EXISTS idx_scribe_audits_run
            ON scribe_audits(run_id, sample_id, audit_model);
        """
    )


FLEURS_COLUMNS = [
    "id",
    "filename",
    "raw_transcription",
    "transcription",
    "characters",
    "num_samples",
    "gender",
]


def _fleurs_repo_file(config: str, split: str, filename: str) -> str:
    return f"data/{config}/{filename.format(split=split)}"


def _download_fleurs_file(dataset: str, path: str) -> Path:
    if dataset != "google/fleurs":
        raise ValueError("TSV/tar ingest currently supports the official google/fleurs repo")
    return Path(hf_hub_download(repo_id=dataset, repo_type="dataset", filename=path))


def _fleurs_file(dataset: str, path: str, source_dir: Path | None) -> Path:
    if source_dir is not None:
        local_path = source_dir.expanduser() / path
        if local_path.exists():
            return local_path
    return _download_fleurs_file(dataset, path)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=FLEURS_COLUMNS, delimiter="\t")
        return [dict(row) for row in reader]


def _sample_exists(
    conn: sqlite3.Connection,
    *,
    dataset: str,
    config: str,
    split: str,
    hf_id: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM fleurs_samples
        WHERE dataset = ? AND config = ? AND split = ? AND hf_id = ?
        """,
        (dataset, config, split, hf_id),
    ).fetchone()
    return row is not None


def _copy_audio_from_tar(
    tar: tarfile.TarFile,
    *,
    filename: str,
    destination: Path,
) -> None:
    member = next(
        (candidate for candidate in tar.getmembers() if Path(candidate.name).name == filename),
        None,
    )
    if member is None:
        raise FileNotFoundError(f"{filename} not found in Google FLEURS audio tar")
    extracted = tar.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"{filename} could not be read from Google FLEURS audio tar")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(extracted.read())


def ingest_samples(
    project: FleursProject,
    *,
    database: Path,
    media_dir: Path,
    dataset: str,
    config: str,
    split: str,
    limit: int,
    source_dir: Path | None = None,
) -> IngestStats:
    """Load official Google FLEURS TSV/audio tar rows into SQLite."""
    conn = connect(database)
    added = 0
    skipped = 0
    scanned = 0
    now = utc_now()
    try:
        ensure_schema(conn)
        tsv_path = _fleurs_file(
            dataset,
            _fleurs_repo_file(config, split, "{split}.tsv"),
            source_dir,
        )
        tar_path = _fleurs_file(
            dataset,
            _fleurs_repo_file(config, split, "audio/{split}.tar.gz"),
            source_dir,
        )
        rows = _read_tsv(tsv_path)
        with tarfile.open(tar_path, mode="r:gz") as tar:
            for row in rows:
                scanned += 1
                hf_id = str(row["id"])
                if _sample_exists(conn, dataset=dataset, config=config, split=split, hf_id=hf_id):
                    skipped += 1
                    continue
                filename = str(row["filename"])
                audio_path = media_dir / config / split / filename
                _copy_audio_from_tar(tar, filename=filename, destination=audio_path)
                ref_text = str(row["transcription"] or row["raw_transcription"])
                normalized_ref_text = project.normalize(ref_text) or ""
                with conn:
                    conn.execute(
                        """
                        INSERT INTO fleurs_samples (
                            dataset, config, split, hf_id, audio_path, sample_rate, num_samples,
                            ref_text, normalized_ref_text, raw_row_json,
                            created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dataset,
                            config,
                            split,
                            hf_id,
                            str(audio_path),
                            16000,
                            int(row["num_samples"]),
                            ref_text,
                            normalized_ref_text,
                            json.dumps(row, ensure_ascii=False, sort_keys=True),
                            now,
                            now,
                        ),
                    )
                added += 1
                if limit and added >= limit:
                    break
    finally:
        conn.close()
    return IngestStats(
        database=str(database),
        dataset=dataset,
        config=config,
        split=split,
        added=added,
        skipped_existing=skipped,
        scanned=scanned,
    )


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
        else score(
            normalized_ref_text,
            normalized_pred_text,
        )
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


def edit_counts(normalized_ref_text: str, normalized_pred_text: str) -> dict[str, int]:
    """Return WER edit operation counts for normalized text."""
    if not normalized_ref_text:
        return {"hits": 0, "substitutions": 0, "insertions": 0, "deletions": 0}
    output = jiwer.process_words(normalized_ref_text, normalized_pred_text)
    return {
        "hits": int(output.hits),
        "substitutions": int(output.substitutions),
        "insertions": int(output.insertions),
        "deletions": int(output.deletions),
    }


def deterministic_audit_category(row: sqlite3.Row) -> tuple[str, str, str] | None:
    """Return category/reason/source when no model call is needed."""
    if not str(row["normalized_ref_text"] or "").strip():
        return (
            "skipped_empty_ref",
            "normalized_ref_text is empty, so the sample was not submitted for transcription",
            "deterministic",
        )
    if row["result_id"] is None:
        return None
    if float(row["wer"] or 0.0) == 0.0:
        return ("exact_match", "WER is 0.0", "deterministic")
    return None


def scribe_audit_payload(row: sqlite3.Row) -> dict[str, Any]:
    """Build the row payload sent to the text audit model."""
    normalized_ref_text = str(row["normalized_ref_text"] or "")
    normalized_pred_text = str(row["normalized_pred_text"] or "")
    return {
        "sample_id": row["sample_id"],
        "hf_id": row["hf_id"],
        "ref_text": row["ref_text"],
        "pred_text": row["pred_text"],
        "normalized_ref_text": normalized_ref_text,
        "normalized_pred_text": normalized_pred_text,
        "wer": row["wer"],
        "cer": row["cer"],
        "error_profile": row["error_profile"],
        "edit_counts": edit_counts(normalized_ref_text, normalized_pred_text),
    }


def create_scribe_audit_fn(model: str) -> AuditFn:
    """Create a Superwhisper-backed audit classifier."""
    client = SuperwhisperClient()

    def _audit(row: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = SCRIBE_AUDIT_PROMPT.format(
            row_json=json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
        )
        result = client.generate_json(
            model,
            [{"role": "user", "content": prompt}],
            schema=SCRIBE_AUDIT_SCHEMA,
            response_format_name="fleurs_scribe_audit",
            max_tokens=256,
        )
        if not isinstance(result, Mapping):
            raise TypeError("audit model returned non-object JSON")
        return result

    return _audit


def pending_audit_rows(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    audit_model: str,
    limit: int,
) -> list[sqlite3.Row]:
    """Return samples needing audit records for a transcription run."""
    conn.row_factory = sqlite3.Row
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    return list(
        conn.execute(
            f"""
            SELECT
                sample.sample_id,
                sample.hf_id,
                sample.ref_text,
                sample.normalized_ref_text,
                result.result_id,
                result.pred_text,
                result.normalized_pred_text,
                result.wer,
                result.cer,
                result.error_profile
            FROM fleurs_samples sample
            LEFT JOIN transcription_results result
              ON result.sample_id = sample.sample_id AND result.run_id = ?
            LEFT JOIN scribe_audits audit
              ON audit.sample_id = sample.sample_id
             AND audit.run_id = ?
             AND audit.audit_model = ?
            WHERE audit.audit_id IS NULL
            ORDER BY sample.sample_id
            {limit_sql}
            """,
            (run_id, run_id, audit_model),
        )
    )


def record_audit(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    sample_id: int,
    audit_model: str,
    category: str,
    reason: str,
    source: str,
    raw_audit: Mapping[str, Any],
    error: str | None = None,
) -> None:
    """Persist one Scribe audit classification."""
    now = utc_now()
    with conn:
        conn.execute(
            """
            INSERT INTO scribe_audits (
                run_id, sample_id, audit_model, category, reason, source,
                raw_audit_json, error, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, sample_id, audit_model) DO UPDATE SET
                category = excluded.category,
                reason = excluded.reason,
                source = excluded.source,
                raw_audit_json = excluded.raw_audit_json,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                sample_id,
                audit_model,
                category,
                reason,
                source,
                json.dumps(dict(raw_audit), ensure_ascii=False, sort_keys=True),
                error,
                now,
                now,
            ),
        )


def audit_scribe_results(
    *,
    database: Path,
    run_name: str,
    model: str,
    limit: int,
    audit: AuditFn | None = None,
) -> AuditStats:
    """Audit a run's reference/prediction differences."""
    conn = connect(database)
    deterministic = 0
    api = 0
    failed = 0
    try:
        run_id = run_id_for_name(conn, run_name)
        rows = pending_audit_rows(conn, run_id=run_id, audit_model=model, limit=limit)
        if audit is None:
            audit = create_scribe_audit_fn(model)
        for row in rows:
            deterministic_result = deterministic_audit_category(row)
            if deterministic_result is not None:
                category, reason, source = deterministic_result
                record_audit(
                    conn,
                    run_id=run_id,
                    sample_id=int(row["sample_id"]),
                    audit_model=model,
                    category=category,
                    reason=reason,
                    source=source,
                    raw_audit={"category": category, "reason": reason},
                )
                deterministic += 1
                continue
            if row["result_id"] is None:
                continue
            payload = scribe_audit_payload(row)
            try:
                classification = audit(payload)
                category = str(classification["category"])
                reason = str(classification["reason"])
                record_audit(
                    conn,
                    run_id=run_id,
                    sample_id=int(row["sample_id"]),
                    audit_model=model,
                    category=category,
                    reason=reason,
                    source="model",
                    raw_audit=classification,
                )
                api += 1
            except Exception as exc:
                record_audit(
                    conn,
                    run_id=run_id,
                    sample_id=int(row["sample_id"]),
                    audit_model=model,
                    category="unclear",
                    reason=str(exc),
                    source="model_error",
                    raw_audit={"category": "unclear", "reason": str(exc)},
                    error=str(exc),
                )
                failed += 1
        pending = len(pending_audit_rows(conn, run_id=run_id, audit_model=model, limit=0))
        return AuditStats(
            str(database),
            run_id,
            run_name,
            model,
            deterministic,
            api,
            failed,
            pending,
        )
    finally:
        conn.close()


def _submit_next(
    pool: ThreadPoolExecutor,
    process: ProcessFn,
    samples: list[dict[str, Any]],
    futures: dict[Future[TranscriptResult], dict[str, Any]],
    index: list[int],
) -> bool:
    if index[0] >= len(samples):
        return False
    sample = samples[index[0]]
    index[0] += 1
    futures[pool.submit(process, Path(sample["audio_path"]))] = sample
    return True


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
        if process is None:
            process = create_process_fn(audio_model(model), key, language=language)
        index = [0]
        futures: dict[Future[TranscriptResult], dict[str, Any]] = {}
        max_in_flight = max(max_workers * 4, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while len(futures) < max_in_flight and _submit_next(
                pool, process, samples, futures, index
            ):
                submitted += 1
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    sample = futures.pop(future)
                    result = future.result()
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
                    while len(futures) < max_in_flight and _submit_next(
                        pool, process, samples, futures, index
                    ):
                        submitted += 1
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
        return RunStats(str(database), run_id, run_name, ok, failed, submitted, pending)
    finally:
        conn.close()


def print_summary(database: Path) -> int:
    """Print table counts and run status rows."""
    conn = connect(database)
    try:
        ensure_schema(conn)
        samples = conn.execute("SELECT COUNT(*) FROM fleurs_samples").fetchone()[0]
        results = conn.execute("SELECT COUNT(*) FROM transcription_results").fetchone()[0]
        print(f"samples\t{samples}")
        print(f"results\t{results}")
        rows = conn.execute(
            """
            SELECT run.run_id, run.run_name, run.model_key, run.language,
                   COUNT(result.result_id) AS results
            FROM transcription_runs run
            LEFT JOIN transcription_results result ON result.run_id = run.run_id
            GROUP BY run.run_id
            ORDER BY run.run_id
            """
        ).fetchall()
        for row in rows:
            print("\t".join("" if value is None else str(value) for value in row))
    finally:
        conn.close()
    return 0


def build_parser(project: FleursProject) -> argparse.ArgumentParser:
    """Build a project CLI parser."""
    parser = argparse.ArgumentParser(
        description=f"Stream Google FLEURS {project.name} samples into SQLite and transcribe them."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--dataset", default=project.dataset)
    ingest.add_argument("--config", default=project.config)
    ingest.add_argument("--split", default=project.split)
    ingest.add_argument("--database", type=Path, default=project.database)
    ingest.add_argument("--media-dir", type=Path, default=project.media_dir)
    ingest.add_argument("--source-dir", type=Path, default=DEFAULT_FLEURS_SOURCE_DIR)
    ingest.add_argument("--limit", type=int, default=5)

    run = subparsers.add_parser("run")
    run.add_argument("--dataset", default=project.dataset)
    run.add_argument("--config", default=project.config)
    run.add_argument("--split", default=project.split)
    run.add_argument("--database", type=Path, default=project.database)
    run.add_argument("--run-name", default=f"{project.config}-scribe-v2")
    run.add_argument("--model", default="scribe-v2")
    run.add_argument("--language", default=project.language)
    run.add_argument("--key", default=None)
    run.add_argument("--limit", type=int, default=5)
    run.add_argument("--max-workers", type=int, default=4)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--database", type=Path, default=project.database)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--database", type=Path, default=project.database)
    audit.add_argument("--run-name", default=f"{project.config}-scribe-v2")
    audit.add_argument("--model", default=DEFAULT_AUDIT_MODEL)
    audit.add_argument("--limit", type=int, default=0)
    return parser


def dispatch(project: FleursProject, argv: list[str] | None = None) -> int:
    """Run a project CLI command."""
    args = build_parser(project).parse_args(argv)
    if args.command == "ingest":
        stats = ingest_samples(
            project,
            database=args.database,
            media_dir=args.media_dir,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            limit=args.limit,
            source_dir=args.source_dir,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        stats = run_transcriptions(
            project,
            database=args.database,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            run_name=args.run_name,
            model=args.model,
            language=args.language,
            key=args.key,
            limit=args.limit,
            max_workers=args.max_workers,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0 if stats.failed == 0 else 1
    if args.command == "summary":
        return print_summary(args.database)
    if args.command == "audit":
        stats = audit_scribe_results(
            database=args.database,
            run_name=args.run_name,
            model=args.model,
            limit=args.limit,
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0 if stats.failed == 0 else 1
    raise SystemExit(f"unknown command: {args.command}")
