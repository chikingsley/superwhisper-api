"""Stream official Google FLEURS TSV/audio rows into SQLite."""
from __future__ import annotations

import csv
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from superwhisper_api.audio.projects.fleurs.db import connect, ensure_schema, utc_now
from superwhisper_api.audio.projects.fleurs.models import IngestStats

if TYPE_CHECKING:
    import sqlite3

    from superwhisper_api.audio.projects.fleurs.models import FleursProject

DEFAULT_FLEURS_SOURCE_DIR = Path("/Volumes/simons-enjoyment/Hugging Face/google-fleurs")

FLEURS_COLUMNS = [
    "id",
    "filename",
    "raw_transcription",
    "transcription",
    "characters",
    "num_samples",
    "gender",
]


class FleursRow(BaseModel):
    """One validated row from a Google FLEURS ``{split}.tsv`` file."""

    id: str
    filename: str
    raw_transcription: str = ""
    transcription: str = ""
    characters: str = ""
    num_samples: int = 0
    gender: str = ""


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


def _read_tsv(path: Path) -> list[FleursRow]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, fieldnames=FLEURS_COLUMNS, delimiter="\t")
        return [FleursRow.model_validate(row) for row in reader]


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
                if _sample_exists(conn, dataset=dataset, config=config, split=split, hf_id=row.id):
                    skipped += 1
                    continue
                audio_path = media_dir / config / split / row.filename
                _copy_audio_from_tar(tar, filename=row.filename, destination=audio_path)
                ref_text = row.transcription or row.raw_transcription
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
                            row.id,
                            str(audio_path),
                            16000,
                            row.num_samples,
                            ref_text,
                            normalized_ref_text,
                            json.dumps(row.model_dump(), ensure_ascii=False, sort_keys=True),
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
