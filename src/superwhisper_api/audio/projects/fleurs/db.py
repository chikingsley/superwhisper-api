"""SQLite connection, schema, and timestamps for the FLEURS projects."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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
    """Create the Google FLEURS sample/run/result/audit tables."""
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
