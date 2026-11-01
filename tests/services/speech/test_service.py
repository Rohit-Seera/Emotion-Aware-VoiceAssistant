"""Behavioral tests for the provider-neutral voice companion service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.result import RuntimeResult
from app.runtime.types import RuntimeTokenUsage
from app.services.speech.contracts import AudioResult, TranscriptionResult
from app.services.speech.exceptions import (
    VoiceSessionAccessError,
    VoiceTurnCancelledError,
)
from app.services.speech.service import VoiceCompanionService


class FakeRuntime:
    """Deterministic runtime double for voice service tests."""

    async def execute(self, **kwargs: object) -> RuntimeResult:
        return RuntimeResult(
            request_id=uuid4(),
            workflow=Workflow.VOICE,
            status=RuntimeStatus.COMPLETED,
            output="I am here for you.",
            metadata={"emotion": {"emotion": "sad"}},
            token_usage=RuntimeTokenUsage(
                prompt_tokens=4,
                completion_tokens=3,
                total_tokens=7,
            ),
        )


class FakeTranscriber:
    """Deterministic STT test double."""

    async def transcribe(
        self,
        audio: bytes,
        *,
        content_type: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        assert audio == b"audio"
        assert content_type == "audio/wav"
        return TranscriptionResult("Hello Aiko", language or "en", 0.4)


class FakeSynthesizer:
    """Deterministic TTS test double."""

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AudioResult:
        del voice, language, emotion
        return AudioResult(data=text.encode(), content_type="audio/mpeg")

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        del voice, language, emotion

        async def chunks() -> AsyncIterator[bytes]:
            yield text.encode()

        return chunks()


@pytest.mark.asyncio
async def test_process_turn_returns_safe_animation_directive() -> None:
    """A completed turn should include constrained frontend animation data."""

    service = VoiceCompanionService(runtime=FakeRuntime())
    session = await service.create_session("user-1")

    turn = await service.process_turn(
        session_id=session.id,
        owner_id="user-1",
        transcript="I feel low today.",
    )

    assert turn.text == "I am here for you."
    assert turn.expression.emotion == "empathetic"
    assert turn.expression.expression == "soft_concern"
    assert turn.expression.lip_sync == "speech"
    assert turn.token_usage.total_tokens == 7


@pytest.mark.asyncio
async def test_audio_stays_in_memory_for_stt_and_tts() -> None:
    """The service should normalize audio without a temporary file contract."""

    service = VoiceCompanionService(
        runtime=FakeRuntime(),
        transcriber=FakeTranscriber(),
        synthesizer=FakeSynthesizer(),
    )

    transcription = await service.transcribe(
        b"audio",
        content_type="audio/wav; charset=binary",
    )
    audio = await service.synthesize("Hello")
    chunks = [chunk async for chunk in service.stream_synthesis("Hello")]

    assert transcription.text == "Hello Aiko"
    assert audio.data == b"Hello"
    assert chunks == [b"Hello"]


@pytest.mark.asyncio
async def test_flac_audio_is_accepted_for_transcription() -> None:
    class FlacTranscriber:
        async def transcribe(
            self,
            audio: bytes,
            *,
            content_type: str,
            language: str | None = None,
        ) -> TranscriptionResult:
            assert audio == b"flac"
            assert content_type == "audio/flac"
            return TranscriptionResult("Flac audio", language or "en", 0.4)

    service = VoiceCompanionService(
        runtime=FakeRuntime(),
        transcriber=FlacTranscriber(),
    )

    result = await service.transcribe(b"flac", content_type="audio/flac")

    assert result.text == "Flac audio"


@pytest.mark.asyncio
async def test_owner_cannot_use_another_users_session() -> None:
    """Session ownership must be checked before a runtime call starts."""

    service = VoiceCompanionService(runtime=FakeRuntime())
    session = await service.create_session("user-1")

    with pytest.raises(VoiceSessionAccessError):
        await service.process_turn(
            session_id=session.id,
            owner_id="user-2",
            transcript="Hello",
        )


@pytest.mark.asyncio
async def test_barge_in_cancels_the_active_turn() -> None:
    """A caller should be able to interrupt their own pending response."""

    started = asyncio.Event()

    class SlowRuntime:
        async def execute(self, **kwargs: object) -> RuntimeResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("The task should have been cancelled.")

    service = VoiceCompanionService(runtime=SlowRuntime())
    session = await service.create_session("user-1")
    task = asyncio.create_task(
        service.process_turn(
            session_id=session.id,
            owner_id="user-1",
            transcript="Please wait.",
        )
    )

    await started.wait()

    assert await service.cancel_turn(session.id, "user-1") is True
    metrics = await service.session_metrics(session.id, "user-1")
    assert metrics["interruptions"] == 1

    with pytest.raises(VoiceTurnCancelledError):
        await task
