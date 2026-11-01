"""Tests for the finish runtime node."""

from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.finish import FinishNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
        output="Hello! How can I help?",
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
async def test_marks_execution_completed(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Finish node should mark execution as completed."""

    node = FinishNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert state.status is RuntimeStatus.COMPLETED


@pytest.mark.asyncio
async def test_preserves_existing_output(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Finish node should preserve the generated output."""

    node = FinishNode()

    await node.execute(
        state,
        context,
    )

    assert state.output == (
        "Hello! How can I help?"
    )


@pytest.mark.asyncio
async def test_initializes_missing_output(
    context: RuntimeContext,
) -> None:
    """Finish node should provide an empty output when needed."""

    state = RuntimeState(
        request_id=context.request_id,
        workflow=context.workflow,
        input="Hello.",
    )

    node = FinishNode()

    await node.execute(
        state,
        context,
    )

    assert state.status is RuntimeStatus.COMPLETED
    assert state.output == ""


@pytest.mark.asyncio
async def test_marks_context_as_finished(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Finish node should mark the context as finished."""

    node = FinishNode()

    await node.execute(
        state,
        context,
    )

    assert context.metadata["finished"] is True


@pytest.mark.asyncio
async def test_returns_same_state_instance(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Finish node should mutate and return the existing state."""

    node = FinishNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state