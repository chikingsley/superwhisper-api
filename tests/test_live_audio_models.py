"""Live audio model checks against real Superwhisper/provider endpoints."""
from __future__ import annotations

from pathlib import Path

import pytest

from superwhisper_api.audio.models import AUDIO_MODELS
from superwhisper_api.audio.transcribe import create_process_fn

FIXTURE_AUDIO = Path("tests/fixtures/superwhisper/1779580688/output.wav")


@pytest.mark.parametrize("spec", tuple(AUDIO_MODELS.values()), ids=list(AUDIO_MODELS))
def test_live_audio_model_returns_raw_response(spec) -> None:
    """Each configured file-transcription model should return raw provider data."""
    process = create_process_fn(spec, key=None)

    result = process(FIXTURE_AUDIO)
    payload = result.as_dict()

    assert "error" not in payload, payload
    assert payload["provider"] == spec.provider
    assert payload["model_key"] == spec.key
    assert payload["model_id"] == spec.model_id
    assert isinstance(payload["raw_response"], dict)
    assert payload["raw_response"]
    assert isinstance(payload["transcript"], str)


@pytest.mark.parametrize("spec", tuple(AUDIO_MODELS.values()), ids=list(AUDIO_MODELS))
def test_live_audio_model_accepts_language(spec) -> None:
    """Passing an explicit language hint should be threaded through to the provider."""
    process = create_process_fn(spec, key=None, language="en")

    result = process(FIXTURE_AUDIO)
    payload = result.as_dict()

    assert "error" not in payload, payload
    assert payload["model_key"] == spec.key
    assert isinstance(payload["transcript"], str)
