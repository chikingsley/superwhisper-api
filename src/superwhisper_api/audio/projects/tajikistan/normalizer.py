"""Tajik text normalization for Google FLEURS curation."""

from __future__ import annotations

from tajiknlp import make_pipeline
from tajiknlp.components.cleaners.text_cleaner import TextCleaner
from tajiknlp.components.normalizers.cyrillic import TajikCyrillicNormalizer

PIPELINE = make_pipeline(
    TextCleaner(),
    TajikCyrillicNormalizer(),
)


def maybe_normalize(text: str) -> str | None:
    """Normalize Tajik Cyrillic text with TajikNLP's Tajik-specific normalizer."""
    doc = PIPELINE(text)
    normalized = str(doc.metadata.get("normalized_text") or "")
    return " ".join(normalized.split())
