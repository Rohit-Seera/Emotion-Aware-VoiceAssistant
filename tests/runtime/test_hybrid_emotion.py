"""Tests for acoustic + lexical emotion fusion."""

import asyncio
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.engines.emotion.acoustic import Wav2Vec2AcousticEmotionDetector
from app.engines.emotion.hybrid import HybridEmotionDetector
from app.engines.emotion.engine import KeywordEmotionDetector
from app.runtime.state import RuntimeState


@pytest.mark.asyncio
async def test_hybrid_prefers_strong_acoustic_result() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "emotion": "sad",
            "confidence": 0.84,
            "source": "acoustic_wav2vec2",
            "model": "test-model",
            "top_emotions": [
                {"emotion": "sad", "confidence": 0.84},
                {"emotion": "neutral", "confidence": 0.10},
            ],
        },
    )

    detector = HybridEmotionDetector(
        acoustic=acoustic,
        lexical=KeywordEmotionDetector(),
    )
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="Tell me about my project.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"wav"},
    )

    assert result["emotion"] == "sad"
    assert result["source"] == "acoustic_wav2vec2"
    assert result["acoustic_available"] is True


@pytest.mark.asyncio
async def test_hybrid_falls_back_to_text_when_audio_is_unavailable() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )

    detector = HybridEmotionDetector(
        acoustic=acoustic,
        lexical=KeywordEmotionDetector(),
    )
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I am really frustrated with this bug.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"bad-wav"},
    )

    assert result["emotion"] == "frustrated"
    assert result["source"] == "text_proxy_fallback"
    assert result["acoustic_available"] is False
    assert result["text"]["emotion"] == "frustrated"


@pytest.mark.asyncio
async def test_hybrid_uses_confident_lexical_signal_when_acoustics_are_neutral() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "emotion": "neutral",
            "confidence": 0.78,
            "model": "test-model",
            "top_emotions": [],
        },
    )
    detector = HybridEmotionDetector(acoustic=acoustic)
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I am frustrated with this bug.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"wav"},
    )

    assert result["emotion"] == "frustrated"
    assert result["confidence"] == 0.9
    assert result["source"] == "hybrid_acoustic_text"


@pytest.mark.asyncio
async def test_hybrid_fuses_agreeing_acoustic_and_lexical_predictions() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "emotion": "neutral",
            "confidence": 0.70,
            "model": "test-model",
            "top_emotions": [],
        },
    )
    detector = HybridEmotionDetector(acoustic=acoustic)
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I have a question about the schedule.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"wav"},
    )

    assert result["emotion"] == "neutral"
    assert result["confidence"] == 0.75
    assert result["source"] == "hybrid_acoustic_text"


@pytest.mark.asyncio
async def test_hybrid_prefers_explicit_positive_text_over_contradictory_acoustics() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "emotion": "sad",
            "confidence": 0.81,
            "model": "test-model",
            "top_emotions": [],
        },
    )
    detector = HybridEmotionDetector(acoustic=acoustic)
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I am happy about the result.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"wav"},
    )

    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.9
    assert result["source"] == "text_override_acoustic"


@pytest.mark.asyncio
async def test_hybrid_uses_text_proxy_when_audio_is_missing() -> None:
    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        return_value=None,
    )
    detector = HybridEmotionDetector(acoustic=acoustic)
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I feel happy about this.",
    )

    result = await detector.detect_emotion_with_context(state, {})

    assert result["emotion"] == "happy"
    assert result["source"] == "text_proxy_fallback"
    assert result["acoustic_available"] is False
    assert result["text"]["emotion"] == "happy"
    acoustic.detect.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_falls_back_when_acoustic_detection_times_out() -> None:
    async def slow_detect(_audio: bytes) -> None:
        await asyncio.sleep(1)

    acoustic = Wav2Vec2AcousticEmotionDetector(
        model_name="test-model",
        enabled=True,
    )
    acoustic.detect = AsyncMock(  # type: ignore[method-assign]
        side_effect=slow_detect,
    )
    detector = HybridEmotionDetector(
        acoustic=acoustic,
        acoustic_timeout=0.01,
    )
    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I am worried about this result.",
    )

    result = await detector.detect_emotion_with_context(
        state,
        {"audio_bytes": b"wav"},
    )

    assert result["emotion"] == "anxious"
    assert result["source"] == "text_proxy_fallback"
    assert result["acoustic_available"] is False
    assert result["text"]["emotion"] == "anxious"
