"""Google FLEURS SQLite transcription workflow, split by job.

Public API is re-exported here so callers use
``superwhisper_api.audio.projects.fleurs`` without depending on the internal
module layout (db / ingest / runs / audit / cli).
"""
from __future__ import annotations

from superwhisper_api.audio.projects.fleurs.audit import audit_scribe_results
from superwhisper_api.audio.projects.fleurs.cli import build_parser, dispatch, print_summary
from superwhisper_api.audio.projects.fleurs.db import connect, ensure_schema, utc_now
from superwhisper_api.audio.projects.fleurs.ingest import ingest_samples
from superwhisper_api.audio.projects.fleurs.models import (
    AuditStats,
    FleursProject,
    IngestStats,
    RunStats,
)
from superwhisper_api.audio.projects.fleurs.runs import (
    ensure_run,
    error_profile,
    pending_samples,
    record_result,
    run_transcriptions,
    score,
)

__all__ = [
    "AuditStats",
    "FleursProject",
    "IngestStats",
    "RunStats",
    "audit_scribe_results",
    "build_parser",
    "connect",
    "dispatch",
    "ensure_run",
    "ensure_schema",
    "error_profile",
    "ingest_samples",
    "pending_samples",
    "print_summary",
    "record_result",
    "run_transcriptions",
    "score",
    "utc_now",
]
