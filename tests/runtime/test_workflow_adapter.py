"""Tests for the runtime workflow adapter."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.adapters.workflow_adapter import (
    RuntimeWorkflowAdapter,
)
from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
    )


@pytest.fixture
def context(state: RuntimeState) -> RuntimeContext:
    """Return a sample runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )


@pytest.mark.asyncio
async def test_adapter_stores_workflow() -> None:
    """Adapter should store the wrapped workflow."""

    workflow = MagicMock()

    adapter = RuntimeWorkflowAdapter(workflow)

    assert adapter.workflow is workflow


@pytest.mark.asyncio
async def test_execute_delegates_to_workflow(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """execute should delegate to the wrapped workflow."""

    expected = RuntimeResult(
        request_id=state.request_id,
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
        output="Hello!",
    )

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value=expected,
    )

    adapter = RuntimeWorkflowAdapter(workflow)

    result = await adapter.execute(
        state,
        context,
    )

    assert result is expected

    workflow.execute.assert_awaited_once_with(
        state,
        context,
    )


@pytest.mark.asyncio
async def test_execute_rejects_invalid_result(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """execute should reject invalid workflow results."""

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value="invalid",
    )

    adapter = RuntimeWorkflowAdapter(workflow)

    with pytest.raises(
        TypeError,
        match="Workflow execute\\(\\) must return RuntimeResult",
    ):
        await adapter.execute(
            state,
            context,
        )


@pytest.mark.asyncio
async def test_stream_delegates_to_workflow(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """stream should delegate to the wrapped workflow."""

    async def workflow_stream(
        received_state: RuntimeState,
        received_context: RuntimeContext,
    ):
        assert received_state is state
        assert received_context is context

        yield "event-1"
        yield "event-2"

    workflow = MagicMock()
    workflow.stream = workflow_stream

    adapter = RuntimeWorkflowAdapter(workflow)

    events = []

    async for event in adapter.stream(
        state,
        context,
    ):
        events.append(event)

    assert events == [
        "event-1",
        "event-2",
    ]


@pytest.mark.asyncio
async def test_stream_rejects_workflow_without_streaming(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """stream should reject workflows without stream support."""

    workflow = MagicMock(spec=["execute"])

    adapter = RuntimeWorkflowAdapter(workflow)

    with pytest.raises(
        TypeError,
        match="Workflow does not support streaming",
    ):
        async for _ in adapter.stream(
            state,
            context,
        ):
            pass