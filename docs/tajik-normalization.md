# Tajik normalization

The Tajikistan Google FLEURS project uses `tajiknlp>=1.2.0` for Tajik Cyrillic
normalization.

`tajiknlp` is currently the best available Tajik-specific normalization choice I
found for this repo: the May 2026 TajikNLP paper describes it as the first
comprehensive open-source Python pipeline for authentic Tajik Cyrillic text and
lists cleaning, normalization, tokenization, morphology, stemming,
lemmatization, and related utilities. The installed package exposes
`TajikCyrillicNormalizer`, which is a direct fit for this project.

Implementation:

```python
from tajiknlp import make_pipeline
from tajiknlp.components.cleaners.text_cleaner import TextCleaner
from tajiknlp.components.normalizers.cyrillic import TajikCyrillicNormalizer

PIPELINE = make_pipeline(
    TextCleaner(),
    TajikCyrillicNormalizer(),
)
```

The package default lowercases text. That matches the scoring use case here:
Google FLEURS Tajik references are lowercase while Scribe may return sentence
case, so case-preserving normalization would inflate WER/CER for casing alone.

References:

- Paper: https://arxiv.org/abs/2605.04583
- Hugging Face community/data namespace: https://huggingface.co/TajikNLPWorld
- PyPI package: https://pypi.org/project/tajiknlp/
