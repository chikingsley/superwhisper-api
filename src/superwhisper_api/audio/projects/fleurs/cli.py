"""Argparse CLI for a Google FLEURS SQLite project (ingest/run/summary/audit)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from superwhisper_api.audio.projects.fleurs.audit import DEFAULT_AUDIT_MODEL, audit_scribe_results
from superwhisper_api.audio.projects.fleurs.db import connect, ensure_schema
from superwhisper_api.audio.projects.fleurs.ingest import DEFAULT_FLEURS_SOURCE_DIR, ingest_samples
from superwhisper_api.audio.projects.fleurs.runs import run_transcriptions

if TYPE_CHECKING:
    from superwhisper_api.audio.projects.fleurs.models import FleursProject


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
        print(json.dumps(stats.model_dump(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        run_stats = run_transcriptions(
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
        print(json.dumps(run_stats.model_dump(), ensure_ascii=False, indent=2))
        return 0 if run_stats.failed == 0 else 1
    if args.command == "summary":
        return print_summary(args.database)
    if args.command == "audit":
        audit_stats = audit_scribe_results(
            database=args.database,
            run_name=args.run_name,
            model=args.model,
            limit=args.limit,
        )
        print(json.dumps(audit_stats.model_dump(), ensure_ascii=False, indent=2))
        return 0 if audit_stats.failed == 0 else 1
    raise SystemExit(f"unknown command: {args.command}")
