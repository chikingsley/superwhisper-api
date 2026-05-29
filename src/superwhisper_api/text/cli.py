"""CLI for generating and auditing text through Superwhisper text model routes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from superwhisper_api.text.client import SuperwhisperClient
from superwhisper_api.text.scribe_audit import _build_scribe_audit_parser, _cmd_scribe_audit

if TYPE_CHECKING:
    from concurrent.futures import Future


@dataclass(frozen=True)
class TextResult:
    """Successful text generation result."""

    prompt: str
    prompt_hash: str
    model: str
    requested_model: str
    returned_model: str
    status_code: int
    text: str
    created_at: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable result dictionary."""
        return {
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "status_code": self.status_code,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class TextFailure:
    """Failed text generation result."""

    prompt: str
    prompt_hash: str
    model: str
    error: str
    attempts: int
    created_at: str

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable failure dictionary."""
        return {
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "model": self.model,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
        }


TextRecord = TextResult | TextFailure
GenerateFn = Callable[[str], TextRecord]


# ─── helpers ───────────────────────────────────────────────────────────────


def _prompt_hash(prompt: str) -> str:
    """Return a stable hash for a prompt."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _created_at() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ─── generate subcommand ───────────────────────────────────────────────────


def _build_generate_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Build the argument parser for the generate subcommand."""
    parser = subparsers.add_parser(
        "generate", help="Generate text via Superwhisper-backed text model routes."
    )
    single = parser.add_mutually_exclusive_group(required=True)
    single.add_argument("--prompt", help="Prompt text for a single generation.")
    single.add_argument("--prompt-file", type=Path, help="File containing one prompt.")
    single.add_argument(
        "--prompts-file",
        type=Path,
        help="Batch file containing one prompt per non-empty line.",
    )
    parser.add_argument("--jsonl", type=Path, help="Output JSONL path for batch results.")
    parser.add_argument("--fail-jsonl", type=Path, help="Output JSONL path for batch failures.")
    parser.add_argument("--model", default="gpt-5.4-mini", help="Text model key.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def _validate_generate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate generate CLI arguments."""
    if args.prompts_file and not args.jsonl:
        parser.error("--jsonl must be specified when using --prompts-file.")
    if not args.prompts_file and args.jsonl:
        parser.error("--jsonl is only valid with --prompts-file.")
    if not args.prompts_file and args.fail_jsonl:
        parser.error("--fail-jsonl is only valid with --prompts-file.")


def _prompt_from_args(args: argparse.Namespace) -> str:
    """Return the single prompt requested by CLI args."""
    if args.prompt is not None:
        return args.prompt
    path = args.prompt_file.expanduser()
    return path.read_text(encoding="utf-8")


def _written_prompt_hashes(path: Path) -> set[str]:
    """Collect prompt_hash values already written to a JSONL file."""
    path = path.expanduser()
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt_hash = payload.get("prompt_hash")
            if isinstance(prompt_hash, str):
                hashes.add(prompt_hash)
    return hashes


def _iter_pending(prompts_file: Path, jsonl: Path) -> Iterator[str]:
    """Yield prompts not already represented in the output JSONL."""
    completed = _written_prompt_hashes(jsonl)
    skipped = 0

    def _pending_prompts() -> Iterator[str]:
        nonlocal skipped
        try:
            with prompts_file.expanduser().open(encoding="utf-8") as handle:
                for line in handle:
                    prompt = line.strip()
                    if not prompt:
                        continue
                    if _prompt_hash(prompt) in completed:
                        skipped += 1
                        continue
                    yield prompt
        finally:
            if skipped:
                print(f"Skipping {skipped} already-written prompts", file=sys.stderr)

    return _pending_prompts()


def _write_text_record(path: Path, record: TextRecord) -> None:
    """Append one text record to a JSONL file."""
    _write_jsonl(path, record.as_dict())


def _create_generate_fn(model: str, max_tokens: int) -> GenerateFn:
    """Create a prompt generation function bound to model settings."""
    client = SuperwhisperClient()

    def _generate(prompt: str) -> TextRecord:
        prompt_hash = _prompt_hash(prompt)
        try:
            response = client.generate(
                model,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return TextResult(
                prompt=prompt,
                prompt_hash=prompt_hash,
                model=model,
                requested_model=response.requested_model,
                returned_model=response.model,
                status_code=response.status_code,
                text=response.text,
                created_at=_created_at(),
            )
        except Exception as exc:
            return TextFailure(
                prompt=prompt,
                prompt_hash=prompt_hash,
                model=model,
                error=str(exc),
                attempts=1,
                created_at=_created_at(),
            )

    return _generate


def _run_single(prompt: str, generate: GenerateFn) -> int:
    """Generate one prompt and print pretty JSON to stdout."""
    record = generate(prompt)
    print(json.dumps(record.as_dict(), indent=2, ensure_ascii=False))
    return 1 if isinstance(record, TextFailure) else 0


def _submit_next(
    pool: ThreadPoolExecutor,
    pending_prompts: Iterator[str],
    futures: set[Future[TextRecord]],
    generate: GenerateFn,
    submitted_ref: list[int],
) -> bool:
    """Pull the next prompt and submit it to the worker pool."""
    try:
        prompt = next(pending_prompts)
    except StopIteration:
        return False
    futures.add(pool.submit(generate, prompt))
    submitted_ref[0] += 1
    return True


def _handle_done(
    done: set[Future[TextRecord]],
    jsonl: Path,
    fail_jsonl: Path | None,
    counters: list[int],
) -> None:
    """Write completed batch records and update counters."""
    for future in done:
        record = future.result()
        if isinstance(record, TextFailure):
            counters[1] += 1
            if fail_jsonl:
                _write_text_record(fail_jsonl, record)
            print(f"FAIL {record.prompt_hash}: {record.error}", file=sys.stderr)
        else:
            counters[0] += 1
            _write_text_record(jsonl, record)
            snippet = record.text[:80].replace("\n", " ")
            print(f"OK {record.prompt_hash}: {snippet}", file=sys.stderr)


def _run_batch(
    prompts_file: Path,
    jsonl: Path,
    fail_jsonl: Path | None,
    max_workers: int,
    generate: GenerateFn,
) -> int:
    """Generate all pending prompts from a batch file."""
    prompts_file = prompts_file.expanduser()
    if not prompts_file.exists():
        print(f"Prompts file not found: {prompts_file}", file=sys.stderr)
        return 1

    pending = _iter_pending(prompts_file, jsonl)
    submitted = [0]
    counters = [0, 0]
    max_in_flight = max(max_workers * 4, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: set[Future[TextRecord]] = set()
        while len(futures) < max_in_flight and _submit_next(
            pool, pending, futures, generate, submitted
        ):
            pass
        if not futures:
            print("No pending prompts.", file=sys.stderr)
            return 0

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            _handle_done(done, jsonl, fail_jsonl, counters)
            while len(futures) < max_in_flight and _submit_next(
                pool, pending, futures, generate, submitted
            ):
                pass

    print(
        f"Done: {counters[0]} ok, {counters[1]} failed, {submitted[0]} submitted",
        file=sys.stderr,
    )
    return 0 if counters[1] == 0 else 1


def _cmd_generate(args: argparse.Namespace) -> int:
    """Handle the generate subcommand."""
    _validate_generate_args(args, argparse.ArgumentParser())
    generate = _create_generate_fn(args.model, args.max_tokens)

    if args.prompts_file:
        return _run_batch(
            args.prompts_file,
            args.jsonl,
            args.fail_jsonl,
            args.max_workers,
            generate,
        )

    return _run_single(_prompt_from_args(args), generate)


# ─── main entry point ──────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="superwhisper-text",
        description="Superwhisper text model CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_generate_parser(subparsers)
    _build_scribe_audit_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for superwhisper-text."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "scribe-audit":
        return _cmd_scribe_audit(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
