"""Persian text normalizer vendored from NVIDIA's Persian FastConformer model card."""

from __future__ import annotations

import string
import unicodedata

SOURCE_REPO = "nvidia/stt_fa_fastconformer_hybrid_large"
SOURCE_FILE = "README.md"
SOURCE_REVISION = "249cf5bf70dda7220a60ddeeecff2f6aad8e1784"
SOURCE_README_SHA256 = "f98ae540031ed90105b887ad3529f412a17ecfd452a5341d904fb4733913ce7e"

SKIP = {
    *string.ascii_letters,
    "=",
    "ā",
    "š",
    "ة",
}

DISCARD = [
    "(خنده)",
    "!",
    '"',
    "#",
    "&",
    "'",
    "(",
    ")",
    ",",
    "-",
    ".",
    ":",
    ";",
    "–",
    "“",
    "”",
    "…",
    "؟",
    "،",
    "؛",
    "ـ",
    "ً",
    "ٌ",
    "َ",
    "ُ",
    "ِ",
    "ّ",
    "ْ",
    "ٔ",
    "«",
    "»",
]

REPLACEMENTS = {
    "أ": "ا",
    "ۀ": "ە",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ے": "ی",
    "ﺒ": "ب",
    "ﻢ": "ﻡ",
    "٬": " ",
    "ە": "ه",
}


def maybe_normalize(text: str) -> str | None:
    """Normalize Persian text, returning None for rows NVIDIA's recipe skips."""
    if set(text) & SKIP:
        return None
    text = " ".join(w for w in text.split() if not w.startswith("#"))
    for lhs, rhs in REPLACEMENTS.items():
        text = text.replace(lhs, rhs)
    for tok in DISCARD:
        text = text.replace(tok, "")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ء", "")
    return " ".join(t for t in text.split() if t)
