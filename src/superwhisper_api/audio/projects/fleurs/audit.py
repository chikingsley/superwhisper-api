"""Classify reference/prediction differences for a transcription run."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jiwer

from superwhisper_api.audio.projects.fleurs.db import connect, utc_now
from superwhisper_api.audio.projects.fleurs.models import AuditStats
from superwhisper_api.audio.projects.fleurs.runs import run_id_for_name
from superwhisper_api.text.client import SuperwhisperClient

if TYPE_CHECKING:
    from pathlib import Path

    from superwhisper_api.audio.projects.fleurs.models import AuditFn

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
            database=str(database),
            run_id=run_id,
            run_name=run_name,
            model=model,
            deterministic=deterministic,
            api=api,
            failed=failed,
            pending=pending,
        )
    finally:
        conn.close()
