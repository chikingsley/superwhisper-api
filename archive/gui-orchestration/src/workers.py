from __future__ import annotations

import argparse
import os
import pwd
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from superwhisper_api.cli import run_job

SUPERWHISPER_APP = Path("/Applications/superwhisper.app")
DEFAULT_WORKERS = ("main", "scribe1", "scribe2", "scribe3", "scribe4")


@dataclass(frozen=True)
class Worker:
    name: str
    user: str

    @property
    def uid(self) -> int:
        return pwd.getpwnam(self.user).pw_uid

    @property
    def home(self) -> Path:
        return Path(pwd.getpwnam(self.user).pw_dir)

    @property
    def db(self) -> Path:
        return self.home / "Library/Application Support/superwhisper/database/superwhisper.sqlite"


WORKERS = {
    "main": Worker("main", "simonpeacocks"),
    "scribe1": Worker("scribe1", "scribe1"),
    "scribe2": Worker("scribe2", "scribe2"),
    "scribe3": Worker("scribe3", "scribe3"),
    "scribe4": Worker("scribe4", "scribe4"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="superwhisper-api-workers",
        description="Preflight and launch the five local Superwhisper worker users.",
    )
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--workers",
        nargs="+",
        choices=sorted(WORKERS),
        default=list(DEFAULT_WORKERS),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def worker_env(worker: Worker) -> list[str]:
    path = os.environ.get("PATH", "")
    return [
        f"HOME={worker.home}",
        f"USER={worker.user}",
        f"LOGNAME={worker.user}",
        f"PATH={path}",
    ]


def as_worker_command(worker: Worker, command: list[str]) -> list[str]:
    return [
        "sudo",
        "launchctl",
        "asuser",
        str(worker.uid),
        "sudo",
        "-u",
        worker.user,
        "env",
        *worker_env(worker),
        *command,
    ]


def output_paths(job_dir: Path, run_id: str, worker: Worker) -> tuple[Path, Path, Path, Path]:
    paths_file = job_dir / f"{run_id}_{worker.name}_paths.txt"
    results = job_dir / "results"
    jsonl = results / f"scribev2.{worker.name}.jsonl"
    failures = results / f"scribev2.{worker.name}.failures.jsonl"
    log = results / f"superwhisper-{run_id}-{worker.name}.log"
    return paths_file, jsonl, failures, log


def preflight_worker(worker: Worker, job_dir: Path, run_id: str) -> list[str]:
    errors: list[str] = []
    paths_file, _, _, log = output_paths(job_dir, run_id, worker)
    results_dir = job_dir / "results"

    if not SUPERWHISPER_APP.exists():
        errors.append(f"missing app: {SUPERWHISPER_APP}")
    if not paths_file.exists():
        errors.append(f"missing paths file: {paths_file}")
    if not results_dir.exists():
        errors.append(f"missing results dir: {results_dir}")
    db_exists = run(["sudo", "test", "-f", str(worker.db)], check=False).returncode == 0
    if not db_exists:
        errors.append(f"missing DB: {worker.db}")
    else:
        stat_result = run(["sudo", "stat", "-f", "%u", str(worker.db)], check=False)
        if stat_result.returncode != 0:
            errors.append(f"cannot stat DB: {worker.db}: {stat_result.stderr.strip()}")
        elif int(stat_result.stdout.strip()) != worker.uid:
            owner = pwd.getpwuid(int(stat_result.stdout.strip())).pw_name
            errors.append(f"DB owner mismatch: {worker.db} owned by {owner}")

    if run(["sudo", "launchctl", "print", f"gui/{worker.uid}"], check=False).returncode != 0:
        errors.append(f"missing gui/{worker.uid} domain for {worker.user}")

    write_check = as_worker_command(worker, ["test", "-w", str(results_dir)])
    if run(write_check, check=False).returncode != 0:
        errors.append(f"{worker.user} cannot write results dir: {results_dir}")

    log.parent.mkdir(parents=True, exist_ok=True)
    return errors


def start_superwhisper(worker: Worker, dry_run: bool) -> None:
    command = as_worker_command(worker, ["open", "-g", "-a", str(SUPERWHISPER_APP)])
    print(" ".join(command))
    if not dry_run:
        run(command)


def open_audio_as_worker(worker: Worker, audio_path: Path) -> None:
    run(as_worker_command(worker, ["open", "-g", "-a", str(SUPERWHISPER_APP), str(audio_path)]))


def run_worker(worker: Worker, job_dir: Path, run_id: str) -> int:
    paths_file, jsonl, failures, log = output_paths(job_dir, run_id, worker)
    log.parent.mkdir(parents=True, exist_ok=True)
    reserve = job_dir / "direct-eleven_reserve_100000_paths.txt"
    skip_paths_files = [reserve] if reserve.exists() else []

    def log_line(message: str) -> None:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    log_line(f"controller thread starting for {worker.name} as {worker.user}")
    try:
        return run_job(
            paths_file,
            jsonl,
            failures,
            db_path=worker.db,
            open_audio_fn=lambda audio_path: open_audio_as_worker(worker, audio_path),
            log=log_line,
            skip_paths_files=skip_paths_files,
        )
    except Exception as exc:
        log_line(f"worker crashed: {exc}")
        raise


def run_controller(selected: list[Worker], job_dir: Path, run_id: str, dry_run: bool) -> int:
    for worker in selected:
        paths_file, _, _, log = output_paths(job_dir, run_id, worker)
        print(f"[{worker.name}] {paths_file} >> {log}")
    if dry_run:
        return 0

    results: dict[str, int] = {}
    failures: dict[str, BaseException] = {}
    threads: list[threading.Thread] = []

    def target(worker: Worker) -> None:
        try:
            results[worker.name] = run_worker(worker, job_dir, run_id)
        except BaseException as exc:
            failures[worker.name] = exc

    for worker in selected:
        thread = threading.Thread(target=target, args=(worker,), name=f"worker-{worker.name}")
        thread.start()
        threads.append(thread)

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("controller interrupted; worker threads will stop with the process", file=sys.stderr)
        return 130

    for name, exc in failures.items():
        print(f"[{name}] failed: {exc}", file=sys.stderr)
    if failures:
        return 1
    return max(results.values(), default=0)


def main() -> int:
    args = build_parser().parse_args()
    job_dir = args.job_dir.expanduser()
    selected = [WORKERS[name] for name in args.workers]

    if os.geteuid() != 0:
        print(
            "superwhisper-api-workers must run as root so one controller can read "
            "all five users' Superwhisper databases. Use sudo.",
            file=sys.stderr,
        )
        return 1

    failed = False
    for worker in selected:
        errors = preflight_worker(worker, job_dir, args.run_id)
        if errors:
            failed = True
            print(f"[{worker.name}] preflight failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
        else:
            print(f"[{worker.name}] preflight ok")

    if failed:
        return 1
    if args.preflight_only:
        return 0

    for worker in selected:
        start_superwhisper(worker, args.dry_run)
    return run_controller(selected, job_dir, args.run_id, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
