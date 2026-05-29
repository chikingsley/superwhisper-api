"""Data shapes for the Google FLEURS SQLite projects."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

NormalizeFn = Callable[[str], "str | None"]
AuditFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class FleursProject:
    """Defaults for one Google FLEURS language project.

    Kept a dataclass (not a Pydantic model) because ``normalize`` holds a
    callable, which is awkward to validate.
    """

    name: str
    dataset: str
    config: str
    split: str
    database: Path
    media_dir: Path
    language: str
    normalize: NormalizeFn


class IngestStats(BaseModel, frozen=True):
    """Summary for streamed sample ingestion."""

    database: str
    dataset: str
    config: str
    split: str
    added: int
    skipped_existing: int
    scanned: int


class RunStats(BaseModel, frozen=True):
    """Summary for a transcription run."""

    database: str
    run_id: int
    run_name: str
    ok: int
    failed: int
    submitted: int
    pending: int


class AuditStats(BaseModel, frozen=True):
    """Summary for Scribe audit classification."""

    database: str
    run_id: int
    run_name: str
    model: str
    deterministic: int
    api: int
    failed: int
    pending: int
