from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


importlib.import_module("app.services.speech.service")
from app.providers.elevenlabs import provider as elevenlabs_provider


@pytest.mark.asyncio
async def test_all_emotional_styles_keep_one_configured_voice(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    async def audio_chunks():
        yield b"audio"

    def convert(**kwargs):
        requests.append(kwargs)
        return audio_chunks()

    client = SimpleNamespace(
        text_to_speech=SimpleNamespace(convert=convert),
    )
    monkeypatch.setattr(
        elevenlabs_provider,
        "AsyncElevenLabs",
        lambda **kwargs: client,
    )
    monkeypatch.setattr(
        elevenlabs_provider,
        "VoiceSettings",
        lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(
        elevenlabs_provider.settings.elevenlabs,
        "api_key",
        "test-key",
    )
    monkeypatch.setattr(
        elevenlabs_provider.settings.elevenlabs,
        "voice_id",
        "one-configured-voice",
    )

    provider = elevenlabs_provider.ElevenLabsTTS()
    await provider.synthesize("I hear you.", voice="different-voice", emotion="empathetic")
    streamed = [
        chunk
        async for chunk in provider.stream(
            "I am here with you.",
            voice="another-voice",
            emotion="calm",
        )
    ]

    assert streamed == [b"audio"]
    assert [request["voice_id"] for request in requests] == [
        "one-configured-voice",
        "one-configured-voice",
    ]
    assert requests[0]["voice_settings"]["similarity_boost"] == 0.90
    assert requests[1]["voice_settings"]["similarity_boost"] == 0.90
    assert requests[0]["voice_settings"]["stability"] >= 0.5
