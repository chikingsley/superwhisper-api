"""Tajikistan Google FLEURS SQLite project CLI."""

from __future__ import annotations

from pathlib import Path

from superwhisper_api.audio.projects.fleurs import FleursProject, dispatch
from superwhisper_api.audio.projects.tajikistan.normalizer import maybe_normalize

PROJECT_DIR = Path(__file__).resolve().parent

PROJECT = FleursProject(
    name="Tajikistan",
    dataset="google/fleurs",
    config="tg_tj",
    split="test",
    database=PROJECT_DIR / "data/google_fleurs.sqlite",
    media_dir=PROJECT_DIR / "data/audio",
    language="tgk",
    normalize=maybe_normalize,
)


def main() -> int:
    """Run the Tajikistan Google FLEURS project CLI."""
    return dispatch(PROJECT)


if __name__ == "__main__":
    raise SystemExit(main())
