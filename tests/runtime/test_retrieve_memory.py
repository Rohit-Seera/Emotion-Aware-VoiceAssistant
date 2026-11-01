"""Tests for the retrieve-memory runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.engines.memory.schemas import MemorySearchResult
from app.runtime.context import RuntimeContext
from app.runtime.nodes.retrieve_memory import RetrieveMemoryNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="What does the user like?",
    )


@pytest.fixture
def context(
    state: RuntimeState,
) -> RuntimeContext:
    """Return a runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "user_id": str(uuid4()),
            "character_id": str(uuid4()),
        },
    )


def make_memory_result() -> MagicMock:
    """Create a mocked memory search result."""

    return MagicMock(spec=MemorySearchResult)


@pytest.mark.asyncio
async def test_retrieves_memories(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should retrieve memories using MemoryEngine."""

    memories = [
        make_memory_result(),
        make_memory_result(),
    ]

    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=memories,
    )

    node = RetrieveMemoryNode(engine)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert result.memories == memories

    engine.search.assert_awaited_once()

    request = engine.search.await_args.args[0]

    assert request.query == (
        "What does the user like?"
    )


@pytest.mark.asyncio
async def test_uses_context_memory_engine(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the engine from runtime context."""

    memories = [
        make_memory_result(),
    ]

    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=memories,
    )

    context.services["memory_engine"] = engine

    node = RetrieveMemoryNode()

    result = await node.execute(
        state,
        context,
    )

    assert result.memories == memories

    engine.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_replaces_existing_memories(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should replace stale memory results."""

    old_memory = make_memory_result()
    new_memory = make_memory_result()

    state.memories = [old_memory]

    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=[new_memory],
    )

    node = RetrieveMemoryNode(engine)

    await node.execute(
        state,
        context,
    )

    assert state.memories == [new_memory]


@pytest.mark.asyncio
async def test_empty_results_clear_memories(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Empty retrieval results should clear existing memories."""

    state.memories = [
        make_memory_result(),
    ]

    engine = MagicMock()
    engine.search = AsyncMock(
        return_value=[],
    )

    node = RetrieveMemoryNode(engine)

    result = await node.execute(
        state,
        context,
    )

    assert result.memories == []


@pytest.mark.asyncio
async def test_missing_memory_engine_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail when no memory engine is configured."""

    node = RetrieveMemoryNode()

    with pytest.raises(
        RuntimeError,
        match="Memory engine has not been configured",
    ):
        await node.execute(
            state,
            context,
        )