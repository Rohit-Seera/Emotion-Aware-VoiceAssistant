"""Tests for the personality runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.personality import PersonalityNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello Weings.",
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
async def test_loads_personality(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should load personality configuration."""

    personality = {
        "tone": "warm",
        "style": "friendly",
        "verbosity": "moderate",
    }

    provider = MagicMock()
    provider.get_personality = AsyncMock(
        return_value=personality,
    )

    node = PersonalityNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["personality"] == personality

    provider.get_personality.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_uses_context_provider(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the provider from runtime context."""

    personality = {
        "tone": "calm",
    }

    provider = MagicMock()
    provider.get_personality = AsyncMock(
        return_value=personality,
    )

    context.services["personality_provider"] = provider

    node = PersonalityNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["personality"] == personality

    provider.get_personality.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_replaces_existing_personality(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should replace an existing personality configuration."""

    context.metadata["personality"] = {
        "tone": "old",
    }

    provider = MagicMock()
    provider.get_personality = AsyncMock(
        return_value={
            "tone": "new",
        },
    )

    node = PersonalityNode(provider)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["personality"] == {
        "tone": "new",
    }


@pytest.mark.asyncio
async def test_empty_personality_is_supported(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Empty personality configuration should be accepted."""

    provider = MagicMock()
    provider.get_personality = AsyncMock(
        return_value={},
    )

    node = PersonalityNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["personality"] == {}


@pytest.mark.asyncio
async def test_missing_provider_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail when no provider is configured."""

    node = PersonalityNode()

    with pytest.raises(
        RuntimeError,
        match="Personality provider has not been configured",
    ):
        await node.execute(
            state,
            context,
        )