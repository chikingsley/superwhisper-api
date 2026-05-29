"""Audio model specifications for cloud transcription providers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AudioProvider = Literal["elevenlabs", "deepgram", "s1", "ultra"]


@dataclass(frozen=True)
class AudioModelSpec:
    """Describes a cloud audio transcription model and its provider."""

    key: str
    provider: AudioProvider
    model_id: str


ELEVENLABS_SCRIBE_V2 = AudioModelSpec(
    key="scribe-v2",
    provider="elevenlabs",
    model_id="scribe_v2",
)
ELEVENLABS_MODELS = (
    ELEVENLABS_SCRIBE_V2,
)

DEEPGRAM_NOVA_2 = AudioModelSpec(
    key="deepgram-nova-2",
    provider="deepgram",
    model_id="nova-2",
)
DEEPGRAM_NOVA_2_MEDICAL = AudioModelSpec(
    key="deepgram-nova-2-medical",
    provider="deepgram",
    model_id="nova-2-medical",
)
DEEPGRAM_NOVA_3 = AudioModelSpec(
    key="deepgram-nova-3",
    provider="deepgram",
    model_id="nova-3",
)
DEEPGRAM_MODELS = (
    DEEPGRAM_NOVA_2,
    DEEPGRAM_NOVA_2_MEDICAL,
    DEEPGRAM_NOVA_3,
)

S1_VOICE = AudioModelSpec(
    key="s1-voice",
    provider="s1",
    model_id="sw-ultra-cloud-v1-east",
)

ULTRA = AudioModelSpec(
    key="ultra",
    provider="ultra",
    model_id="sw-ultra-cloud-v1-east",
)
SUPERWHISPER_AUDIO_MODELS = (
    S1_VOICE,
    ULTRA,
)

AUDIO_MODELS = {
    model.key: model
    for model in (
        *ELEVENLABS_MODELS,
        *DEEPGRAM_MODELS,
        *SUPERWHISPER_AUDIO_MODELS,
    )
}


def audio_model(name: str) -> AudioModelSpec:
    """Resolve a canonical audio model key to its AudioModelSpec."""
    try:
        return AUDIO_MODELS[name]
    except KeyError as exc:
        known = ", ".join(sorted(AUDIO_MODELS))
        raise ValueError(f"Unknown audio model {name!r}. Known models: {known}") from exc
