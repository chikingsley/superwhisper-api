"""Persian Google FLEURS SQLite project CLI."""

from __future__ import annotations

from pathlib import Path

from superwhisper_api.audio.projects.fleurs import FleursProject, dispatch
from superwhisper_api.audio.projects.persian.normalizer import maybe_normalize

PROJECT_DIR = Path(__file__).resolve().parent

PROJECT = FleursProject(
    name="Persian",
    dataset="google/fleurs",
    config="fa_ir",
    split="test",
    database=PROJECT_DIR / "data/google_fleurs.sqlite",
    media_dir=PROJECT_DIR / "data/audio",
    language="fas",
    normalize=maybe_normalize,
)


def main() -> int:
    """Run the Persian Google FLEURS project CLI."""
    return dispatch(PROJECT)


if __name__ == "__main__":
    raise SystemExit(main())
