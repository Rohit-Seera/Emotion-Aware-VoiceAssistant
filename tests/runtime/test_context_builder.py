"""Tests for the context-builder runtime node."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.context_builder import ContextBuilderNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello Weings.",
        session_id=uuid4(),
        memories=[
            {"content": "User likes Python."},
        ],
        metadata={
            "emotion": "happy",
        },
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
            "existing": True,
        },
    )


@pytest.mark.asyncio
async def test_execute_returns_same_state(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Context building should not replace runtime state."""

    node = ContextBuilderNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state


@pytest.mark.asyncio
async def test_builds_context(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should build structured runtime context."""

    node = ContextBuilderNode()

    await node.execute(
        state,
        context,
    )

    built = context.metadata["built_context"]

    assert built["input"] == "Hello Weings."
    assert built["session_id"] == state.session_id
    assert built["workflow"] is Workflow.CHAT
    assert built["memories"] == state.memories
    assert built["metadata"] == state.metadata


@pytest.mark.asyncio
async def test_preserves_existing_context_metadata(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Existing context metadata should not be removed."""

    node = ContextBuilderNode()

    await node.execute(
        state,
        context,
    )

    assert context.metadata["existing"] is True
    assert "built_context" in context.metadata


def test_execute_context_returns_expected_shape(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """execute_context should return the structured context."""

    node = ContextBuilderNode()

    result = node.execute_context(
        state,
        context,
    )

    assert set(result.keys()) == {
        "input",
        "session_id",
        "workflow",
        "memories",
        "metadata",
    }


def test_execute_context_copies_collections(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Context collections should not share mutable containers."""

    node = ContextBuilderNode()

    result = node.execute_context(
        state,
        context,
    )

    assert result["memories"] is not state.memories
    assert result["metadata"] is not state.metadata


@pytest.mark.asyncio
async def test_empty_memories_are_supported(
    context: RuntimeContext,
) -> None:
    """Context building should work without memories."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
    )

    node = ContextBuilderNode()

    await node.execute(
        state,
        context,
    )

    built = context.metadata["built_context"]

    assert built["memories"] == []

@pytest.mark.asyncio
async def test_raw_audio_is_not_copied_into_built_context(
    context: RuntimeContext,
) -> None:
    """Transient microphone bytes should not be duplicated in built context."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
        metadata={
            "audio_bytes": b"raw-microphone-data",
            "emotion": {"detected": "happy"},
        },
    )

    node = ContextBuilderNode()

    result = node.execute_context(state, context)

    assert "audio_bytes" not in result["metadata"]
    assert result["metadata"]["emotion"] == {"detected": "happy"}
