"""Audit scribe transcripts against canonical reference text.

Classifies normalized Persian ASR differences for dataset curation. Exposed as
the ``scribe-audit`` subcommand of ``superwhisper-text``.
"""
from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from superwhisper_api.text.client import SuperwhisperClient, json_schema_response_format
from superwhisper_api.text.models import model_spec

if TYPE_CHECKING:
    import argparse
    from collections.abc import Iterator


# ─── scribe-audit prompt and schema ──────────────────────────────────────────

SCRIBE_AUDIT_PROMPT_TEMPLATE = """You classify normalized Persian ASR text
differences for dataset curation.

For the input row below, compare `normalized_reference` and `normalized_scribe`.
Return JSON with exactly these keys:
sample_id, job_order, category, difference_description, likely_cause, suggested_action.

Rules:
- `category` must be exactly one of:
  exact_match,
  near_match,
  punctuation_or_orthography_only,
  speaker_label_or_annotation,
  boundary_mismatch,
  extra_speech,
  omitted_speech,
  wrong_segment,
  language_mismatch,
  script_mismatch,
  non_speech_annotation,
  number_or_symbol_mismatch,
  named_entity_mismatch,
  content_mismatch,
  low_confidence_unclear.
- `difference_description` must be specific to the strings, not generic.
- Use English for the category/description fields.
- Use `exact_match` when `normalized_reference` equals `normalized_scribe`.
- Use `near_match` for small non-semantic lexical differences with no apparent
  missing or extra speech.
- Use `punctuation_or_orthography_only` only for punctuation, spacing,
  diacritics, Arabic/Persian character variants, or spelling-convention
  differences that do not change words.
- Use `boundary_mismatch` only when one side appears to contain adjacent
  same-recording speech at the beginning or end.
- Use `extra_speech` when Scribe includes speech absent from the reference.
- Use `omitted_speech` when Scribe omits speech present in the reference.
- Use `wrong_segment` only when the strings appear to be different utterances,
  not merely partial overlap.
- Use `language_mismatch` when the hypothesis is a different spoken language.
- Use `script_mismatch` when the content is a transliteration or wrong writing
  system for otherwise related speech.
- Use `non_speech_annotation` when either side contains bracketed non-speech
  markers, labels, or annotations such as `[سکوت]`, `[صدای سکوت]`, or
  `[صدای محیط]`.
- Use `number_or_symbol_mismatch` for numbers, dates, currencies, symbols,
  units, or abbreviations.
- Use `named_entity_mismatch` for person, place, organization, title, or
  proper-name changes.
- Use `content_mismatch` for ordinary meaning-changing word substitutions.
- Use `low_confidence_unclear` when text alone is insufficient to distinguish
  boundary, wrong segment, or content mismatch.
- Do not add markdown, prose, or code fences.

Input row:
{row_json}
"""

SCRIBE_AUDIT_CATEGORIES = {
    "exact_match",
    "near_match",
    "punctuation_or_orthography_only",
    "speaker_label_or_annotation",
    "boundary_mismatch",
    "extra_speech",
    "omitted_speech",
    "wrong_segment",
    "language_mismatch",
    "script_mismatch",
    "non_speech_annotation",
    "number_or_symbol_mismatch",
    "named_entity_mismatch",
    "content_mismatch",
    "low_confidence_unclear",
}

SCRIBE_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_id": {"type": "string"},
        "job_order": {"type": "integer"},
        "category": {
            "type": "string",
            "enum": sorted(SCRIBE_AUDIT_CATEGORIES),
        },
        "difference_description": {"type": "string"},
        "likely_cause": {"type": "string"},
        "suggested_action": {"type": "string"},
    },
    "required": [
        "sample_id",
        "job_order",
        "category",
        "difference_description",
        "likely_cause",
        "suggested_action",
    ],
    "additionalProperties": False,
}


# ─── helpers ───────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    path = path.expanduser()
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if stripped:
                yield json.loads(stripped)


def _row_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return a stable key for a row."""
    return (str(row.get("sample_id", "")), int(row.get("job_order", 0)))


def _completed_row_keys(path: Path) -> set[tuple[str, int]]:
    """Collect row keys already written to an audit output JSONL."""
    keys: set[tuple[str, int]] = set()
    for row in _read_jsonl(path):
        keys.add(_row_key(row))
    return keys


def _build_audit_prompt(row: dict[str, Any]) -> str:
    """Build the classification prompt for a single row."""
    payload = {
        "sample_id": row.get("sample_id", ""),
        "job_order": row.get("job_order", 0),
        "normalized_reference": row.get("normalized_reference", ""),
        "normalized_scribe": row.get("normalized_scribe", ""),
        "wer": row.get("wer", 0.0),
        "cer": row.get("cer", 0.0),
    }
    return SCRIBE_AUDIT_PROMPT_TEMPLATE.format(row_json=json.dumps(payload, ensure_ascii=False))


def _merge_audit_result(
    source_row: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    """Merge classification fields into the source row."""
    return {
        **source_row,
        "difference_category": classification["category"],
        "difference_description": classification["difference_description"],
        "likely_cause": classification["likely_cause"],
        "suggested_action": classification["suggested_action"],
    }


def _audit_response_format(model: str) -> dict[str, Any] | None:
    """Build a JSON schema response format if the model supports it."""
    spec = model_spec(model)
    if spec.supports_response_format:
        return json_schema_response_format("scribe_audit", SCRIBE_AUDIT_SCHEMA)
    return None


def _collect_pending_rows(
    input_path: Path, skip_keys: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Read input JSONL and return rows not already in skip_keys."""
    pending: list[dict[str, Any]] = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if _row_key(row) not in skip_keys:
            pending.append(row)
    return pending


def _auto_exact_match(row: dict[str, Any]) -> dict[str, Any]:
    """Tag a row with WER==0 as exact_match without an API call."""
    return {
        **row,
        "difference_category": "exact_match",
        "difference_description": "exact match",
        "likely_cause": "n/a",
        "suggested_action": "n/a",
    }


def _audit_worker(
    row: dict[str, Any],
    client: SuperwhisperClient,
    model: str,
    response_format: dict[str, Any] | None,
    out_handle,
    write_lock: threading.Lock,
    counters_lock: threading.Lock,
    total_pending: int,
    counters: list[int],
) -> None:
    """Process a single audit row."""
    wer = float(row.get("wer", 1.0))
    if wer == 0.0:
        merged = _auto_exact_match(row)
        line = json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n"
        with write_lock:
            out_handle.write(line)
            out_handle.flush()
        with counters_lock:
            counters[0] += 1
        return
    ok = _process_audit_row(
        row, client, model, response_format, out_handle, write_lock
    )
    with counters_lock:
        if ok:
            counters[0] += 1
            if counters[0] % 10 == 0:
                print(
                    f"Processed {counters[0]}/{total_pending} rows...",
                    file=sys.stderr,
                )
        else:
            counters[1] += 1


def _process_audit_row(
    row: dict[str, Any],
    client: SuperwhisperClient,
    model: str,
    response_format: dict[str, Any] | None,
    out_handle,
    write_lock: threading.Lock | None = None,
    max_retries: int = 3,
) -> bool:
    """Classify a single row and write the merged result."""
    prompt = _build_audit_prompt(row)
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            if response_format:
                classification = client.generate_json(
                    model,
                    [{"role": "user", "content": prompt}],
                    schema=SCRIBE_AUDIT_SCHEMA,
                    response_format_name="scribe_audit",
                )
            else:
                response = client.generate(
                    model,
                    [{"role": "user", "content": prompt}],
                )
                classification = response.parsed()

            category = str(classification.get("category", ""))
            if category not in SCRIBE_AUDIT_CATEGORIES:
                print(
                    f"WARN invalid category {category!r} for "
                    f"{_row_key(row)} — treating as failure",
                    file=sys.stderr,
                )
                return False

            merged = _merge_audit_result(row, classification)
            line = json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n"
            if write_lock:
                with write_lock:
                    out_handle.write(line)
                    out_handle.flush()
            else:
                out_handle.write(line)
                out_handle.flush()
            return True

        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries:
                print(
                    f"RETRY {_row_key(row)} (attempt {attempt + 2}/{max_retries + 1}): {exc}",
                    file=sys.stderr,
                )
                continue
            break

    print(f"FAIL {_row_key(row)}: {last_error}", file=sys.stderr)
    return False


def _cmd_scribe_audit(args: argparse.Namespace) -> int:
    """Handle the scribe-audit subcommand with parallel workers."""
    input_path = args.input.expanduser()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (args.output or input_path.with_suffix(".audit.jsonl")).expanduser()
    skip_keys = _completed_row_keys(output_path) if args.resume else set()
    if skip_keys:
        print(f"Resuming: skipping {len(skip_keys)} already-audited rows", file=sys.stderr)

    client = SuperwhisperClient()
    response_format = _audit_response_format(args.model)

    pending_rows = _collect_pending_rows(input_path, skip_keys)
    if not pending_rows:
        print("No pending rows to process.", file=sys.stderr)
        return 0

    if args.tag_exact_matches:
        exact_rows = [row for row in pending_rows if float(row.get("wer", 1.0)) == 0.0]
        total = len(exact_rows)
        print(
            f"Fast-tagging {total} exact-match rows (no API calls)...",
            file=sys.stderr,
        )
        mode = "a" if args.resume else "w"
        with output_path.open(mode, encoding="utf-8") as out_handle:
            for i, row in enumerate(exact_rows, 1):
                merged = _auto_exact_match(row)
                line = json.dumps(merged, ensure_ascii=False, sort_keys=True) + "\n"
                out_handle.write(line)
                if i % 1000 == 0:
                    out_handle.flush()
                    print(f"Tagged {i}/{total} rows...", file=sys.stderr)
            out_handle.flush()
        print(f"Done: {total} exact-match rows tagged.", file=sys.stderr)
        return 0

    if args.limit:
        pending_rows = pending_rows[: args.limit]

    total_pending = len(pending_rows)
    print(
        f"Processing {total_pending} rows with {args.max_workers} workers...",
        file=sys.stderr,
    )

    counters = [0, 0]  # processed, failed
    skipped = len(skip_keys)
    write_lock = threading.Lock()
    counters_lock = threading.Lock()

    mode = "a" if args.resume else "w"
    with (
        output_path.open(mode, encoding="utf-8") as out_handle,
        ThreadPoolExecutor(max_workers=args.max_workers) as pool,
    ):
        futures = [
            pool.submit(
                _audit_worker,
                row,
                client,
                args.model,
                response_format,
                out_handle,
                write_lock,
                counters_lock,
                total_pending,
                counters,
            )
            for row in pending_rows
        ]
        for future in futures:
            future.result()

    print(
        f"Done: {counters[0]} processed, {skipped} skipped, {counters[1]} failed",
        file=sys.stderr,
    )
    return 0 if counters[1] == 0 else 1


def _build_scribe_audit_parser(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Build the argument parser for the scribe-audit subcommand."""
    parser = subparsers.add_parser(
        "scribe-audit",
        help="Audit scribe transcripts against canonical reference text.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Joined JSONL input path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <input_stem>.audit.jsonl).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Text model key (default: gpt-5.4-mini).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the output file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of rows to process (default: 0 = all).",
    )
    parser.add_argument(
        "--tag-exact-matches",
        action="store_true",
        help="Fast-pass: tag all WER==0 rows as exact_match, no API calls.",
    )
    return parser
