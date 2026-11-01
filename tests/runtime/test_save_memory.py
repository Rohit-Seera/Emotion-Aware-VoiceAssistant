"""Tests for the save-memory runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.engines.memory.schemas import MemoryRead
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from app.runtime.context import RuntimeContext
from app.runtime.nodes.save_memory import SaveMemoryNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="I love Python.",
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
        metadata={
            "user_id": str(uuid4()),
            "character_id": str(uuid4()),
            "memory_to_save": {
                "content": "User loves Python.",
                "memory_type": MemoryType.PREFERENCE,
                "source": MemorySource.CONVERSATION,
                "importance": 0.8,
                "metadata": {
                    "topic": "programming",
                },
            },
        },
    )


def make_memory() -> MagicMock:
    """Create a mocked saved memory."""

    memory = MagicMock(
        spec=MemoryRead,
    )

    memory.id = uuid4()
    memory.content = "User loves Python."
    memory.memory_type = MemoryType.PREFERENCE
    memory.source = MemorySource.CONVERSATION
    memory.status = MemoryStatus.ACTIVE
    memory.importance = 0.8

    return memory


@pytest.mark.asyncio
async def test_saves_memory(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should persist the configured memory."""

    saved_memory = make_memory()

    engine = MagicMock()
    engine.create = AsyncMock(
        return_value=saved_memory,
    )

    node = SaveMemoryNode(engine)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["saved_memory"] is saved_memory

    engine.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_creates_correct_memory_schema(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should construct the correct MemoryCreate."""

    engine = MagicMock()
    engine.create = AsyncMock(
        return_value=make_memory(),
    )

    node = SaveMemoryNode(engine)

    await node.execute(
        state,
        context,
    )

    memory = engine.create.await_args.args[0]

    assert memory.content == "User loves Python."
    assert memory.memory_type is MemoryType.PREFERENCE
    assert memory.source is MemorySource.CONVERSATION
    assert memory.importance == 0.8
    assert memory.metadata == {
        "topic": "programming",
    }


@pytest.mark.asyncio
async def test_uses_context_memory_engine(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the memory engine from context."""

    saved_memory = make_memory()

    engine = MagicMock()
    engine.create = AsyncMock(
        return_value=saved_memory,
    )

    context.services["memory_engine"] = engine

    node = SaveMemoryNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["saved_memory"] is saved_memory

    engine.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_memory_does_nothing(
    state: RuntimeState,
) -> None:
    """Node should do nothing without a memory request."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )

    engine = MagicMock()

    node = SaveMemoryNode(engine)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["saved_memory"] is None

    engine.create.assert_not_called()


@pytest.mark.asyncio
async def test_missing_engine_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail without a memory engine."""

    node = SaveMemoryNode()

    with pytest.raises(
        RuntimeError,
        match="Memory engine has not been configured",
    ):
        await node.execute(
            state,
            context,
        )


@pytest.mark.asyncio
async def test_invalid_memory_content_raises(
    state: RuntimeState,
) -> None:
    """Node should reject empty memory content."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "memory_to_save": {
                "content": "   ",
                "memory_type": MemoryType.PREFERENCE,
            },
        },
    )

    engine = MagicMock()
    engine.create = AsyncMock()

    node = SaveMemoryNode(engine)

    with pytest.raises(
        ValueError,
        match="Memory content must be a non-empty string",
    ):
        await node.execute(
            state,
            context,
        )

    engine.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_memory_type_raises(
    state: RuntimeState,
) -> None:
    """Node should reject invalid memory types."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "memory_to_save": {
                "content": "User likes Python.",
                "memory_type": "invalid",
            },
        },
    )

    engine = MagicMock()
    engine.create = AsyncMock()

    node = SaveMemoryNode(engine)

    with pytest.raises(
        ValueError,
        match="Invalid memory type",
    ):
        await node.execute(
            state,
            context,
        )

    engine.create.assert_not_awaited()