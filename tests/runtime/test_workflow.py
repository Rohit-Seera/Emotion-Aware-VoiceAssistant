"""Tests for the runtime workflow."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState
from app.runtime.workflow import RuntimeWorkflow


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
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


def make_result(
    state: RuntimeState,
) -> RuntimeResult:
    """Create a sample runtime result."""

    return RuntimeResult(
        request_id=state.request_id,
        workflow=state.workflow,
        status=RuntimeStatus.COMPLETED,
        output="Hello from Weings.",
    )


@pytest.mark.asyncio
async def test_execute_delegates_to_adapter(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Workflow should delegate execution to its adapter."""

    result = make_result(state)

    adapter = MagicMock()
    adapter.invoke = AsyncMock(
        return_value=result,
    )

    workflow = RuntimeWorkflow(adapter)

    returned = await workflow.execute(
        state,
        context,
    )

    assert returned is result

    adapter.invoke.assert_awaited_once_with(
        state,
        context,
    )


@pytest.mark.asyncio
async def test_execute_returns_runtime_result(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Workflow should return the adapter's RuntimeResult."""

    result = make_result(state)

    adapter = MagicMock()
    adapter.invoke = AsyncMock(
        return_value=result,
    )

    workflow = RuntimeWorkflow(adapter)

    returned = await workflow.execute(
        state,
        context,
    )

    assert isinstance(
        returned,
        RuntimeResult,
    )


@pytest.mark.asyncio
async def test_stream_delegates_to_adapter(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Workflow should stream events from its adapter."""

    async def adapter_stream(
        received_state: RuntimeState,
        received_context: RuntimeContext,
    ):
        yield {
            "type": "node_started",
            "node": "llm",
        }

        yield {
            "type": "node_finished",
            "node": "llm",
        }

    adapter = MagicMock()
    adapter.stream = adapter_stream

    workflow = RuntimeWorkflow(adapter)

    events = [
        event
        async for event in workflow.stream(
            state,
            context,
        )
    ]

    assert events == [
        {
            "type": "node_started",
            "node": "llm",
        },
        {
            "type": "node_finished",
            "node": "llm",
        },
    ]


@pytest.mark.asyncio
async def test_stream_passes_state_and_context(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Workflow should pass state and context to the adapter."""

    received = {}

    async def adapter_stream(
        received_state: RuntimeState,
        received_context: RuntimeContext,
    ):
        received["state"] = received_state
        received["context"] = received_context

        yield "event"

    adapter = MagicMock()
    adapter.stream = adapter_stream

    workflow = RuntimeWorkflow(adapter)

    events = [
        event
        async for event in workflow.stream(
            state,
            context,
        )
    ]

    assert events == ["event"]
    assert received["state"] is state
    assert received["context"] is context


def test_workflow_exposes_adapter() -> None:
    """Workflow should expose its configured adapter."""

    adapter = MagicMock()

    workflow = RuntimeWorkflow(adapter)

    assert workflow.adapter is adapter