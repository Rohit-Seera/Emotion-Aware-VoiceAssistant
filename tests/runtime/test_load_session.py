"""Tests for the load-session runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.load_session import LoadSessionNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state():
    """Return a runtime state with a session."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        session_id=uuid4(),
    )


@pytest.fixture
def context(state: RuntimeState) -> RuntimeContext:
    """Return a runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )


@pytest.mark.asyncio
async def test_loads_session(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should load session data."""

    session = {
        "user_name": "Rohit",
        "conversation_id": "abc",
    }

    loader = MagicMock()
    loader.load = AsyncMock(
        return_value=session,
    )

    node = LoadSessionNode(loader)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert result.metadata["session"] == session

    loader.load.assert_awaited_once_with(
        state.session_id,
    )


@pytest.mark.asyncio
async def test_uses_context_session_loader(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the context loader when no loader is injected."""

    session = {
        "user_name": "Rohit",
    }

    loader = MagicMock()
    loader.load = AsyncMock(
        return_value=session,
    )

    context.services["session_loader"] = loader

    node = LoadSessionNode()

    result = await node.execute(
        state,
        context,
    )

    assert result.metadata["session"] == session

    loader.load.assert_awaited_once_with(
        state.session_id,
    )


@pytest.mark.asyncio
async def test_no_session_id_returns_state_unchanged(
    context: RuntimeContext,
) -> None:
    """Node should do nothing when no session exists."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
    )

    loader = MagicMock()
    loader.load = AsyncMock()

    node = LoadSessionNode(loader)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert result.metadata == {}

    loader.load.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_session_returns_state(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Missing sessions should not fail the runtime."""

    loader = MagicMock()
    loader.load = AsyncMock(
        return_value=None,
    )

    node = LoadSessionNode(loader)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert result.metadata == {}

    loader.load.assert_awaited_once_with(
        state.session_id,
    )


@pytest.mark.asyncio
async def test_missing_loader_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail clearly when no loader is configured."""

    node = LoadSessionNode()

    with pytest.raises(
        RuntimeError,
        match="Session loader has not been configured",
    ):
        await node.execute(
            state,
            context,
        )
        