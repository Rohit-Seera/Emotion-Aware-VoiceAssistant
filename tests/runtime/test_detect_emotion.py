"""Tests for the emotion detection runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.detect_emotion import DetectEmotionNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="I'm really happy today!",
    )


@pytest.fixture
def context(
    state: RuntimeState,
) -> RuntimeContext:
    """Return a sample runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )


@pytest.mark.asyncio
async def test_detects_emotion(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should detect and store emotion."""

    emotion = {
        "label": "happy",
        "confidence": 0.94,
    }

    provider = MagicMock()
    provider.detect_emotion = AsyncMock(
        return_value=emotion,
    )

    node = DetectEmotionNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["emotion"] == emotion

    provider.detect_emotion.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_uses_context_provider(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the provider from runtime context."""

    emotion = {
        "label": "calm",
        "confidence": 0.81,
    }

    provider = MagicMock()
    provider.detect_emotion = AsyncMock(
        return_value=emotion,
    )

    context.services["emotion_provider"] = provider

    node = DetectEmotionNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["emotion"] == emotion

    provider.detect_emotion.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_replaces_existing_emotion(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should replace stale emotion data."""

    context.metadata["emotion"] = {
        "label": "sad",
    }

    provider = MagicMock()
    provider.detect_emotion = AsyncMock(
        return_value={
            "label": "happy",
        },
    )

    node = DetectEmotionNode(provider)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["emotion"] == {
        "label": "happy",
    }


@pytest.mark.asyncio
async def test_empty_emotion_is_supported(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Empty emotion results should be accepted."""

    provider = MagicMock()
    provider.detect_emotion = AsyncMock(
        return_value={},
    )

    node = DetectEmotionNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["emotion"] == {}


@pytest.mark.asyncio
async def test_missing_provider_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail when no provider is configured."""

    node = DetectEmotionNode()

    with pytest.raises(
        RuntimeError,
        match="Emotion provider has not been configured",
    ):
        await node.execute(
            state,
            context,
        )