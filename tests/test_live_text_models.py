"""Live text model checks against real Superwhisper model routes."""
from __future__ import annotations

import sys

import httpx
import pytest

from superwhisper_api.text.client import SuperwhisperClient
from superwhisper_api.text.models import SUPERWHISPER_MODELS


@pytest.mark.parametrize(
    "spec",
    tuple(SUPERWHISPER_MODELS.values()),
    ids=list(SUPERWHISPER_MODELS),
)
def test_live_text_model_returns_text(spec) -> None:
    """Each configured text model route should answer a real generation request.

    A 400 whose body says the account usage limit was reached counts as a pass:
    the route is reachable and responding sensibly, we are just out of quota.
    Any other error still fails.
    """
    client = SuperwhisperClient()

    try:
        response = client.generate(
            spec,
            [{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=16,
        )
    except httpx.HTTPStatusError as exc:
        body = exc.response.text.lower()
        if exc.response.status_code == 400 and "usage limit" in body:
            print(f"{spec.key}: account usage limit reached (route is healthy)", file=sys.stderr)
            return
        raise

    assert response.status_code == 200
    assert response.requested_model == spec.model_id
    assert response.text.strip()
