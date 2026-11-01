"""Tests for the relationship runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.relationship import RelationshipNode
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
async def test_loads_relationship(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should load relationship context."""

    relationship = {
        "level": 3,
        "familiarity": 0.7,
        "interaction_count": 25,
    }

    provider = MagicMock()
    provider.get_relationship = AsyncMock(
        return_value=relationship,
    )

    node = RelationshipNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["relationship"] == relationship

    provider.get_relationship.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_uses_context_provider(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the provider from runtime context."""

    relationship = {
        "level": 2,
    }

    provider = MagicMock()
    provider.get_relationship = AsyncMock(
        return_value=relationship,
    )

    context.services["relationship_provider"] = provider

    node = RelationshipNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["relationship"] == relationship

    provider.get_relationship.assert_awaited_once_with(
        state,
    )


@pytest.mark.asyncio
async def test_replaces_existing_relationship(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should replace existing relationship context."""

    context.metadata["relationship"] = {
        "level": 1,
    }

    provider = MagicMock()
    provider.get_relationship = AsyncMock(
        return_value={
            "level": 4,
        },
    )

    node = RelationshipNode(provider)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["relationship"] == {
        "level": 4,
    }


@pytest.mark.asyncio
async def test_empty_relationship_is_supported(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Empty relationship context should be accepted."""

    provider = MagicMock()
    provider.get_relationship = AsyncMock(
        return_value={},
    )

    node = RelationshipNode(provider)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["relationship"] == {}


@pytest.mark.asyncio
async def test_missing_provider_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail when no provider is configured."""

    node = RelationshipNode()

    with pytest.raises(
        RuntimeError,
        match="Relationship provider has not been configured",
    ):
        await node.execute(
            state,
            context,
        )