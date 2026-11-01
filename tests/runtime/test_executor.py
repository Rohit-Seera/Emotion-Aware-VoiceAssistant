"""Tests for the runtime executor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.context import RuntimeContext
from app.runtime.exceptions import (
    RuntimeCancelledError,
    RuntimeExecutionError,
    RuntimeTimeoutError,
)
from app.runtime.executor import RuntimeExecutor
from app.runtime.metrics import RuntimeMetrics
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
        status=RuntimeStatus.COMPLETED,
        output="Hello from Weings.",
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
        metadata={
            "provider": "openai",
        },
        token_usage=RuntimeTokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


@pytest.mark.asyncio
async def test_executor_returns_runtime_result(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should return the workflow result."""

    expected = make_result(state)

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value=expected,
    )

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
    )

    result = await executor.execute(
        state,
        context,
    )

    assert result is expected

    workflow.execute.assert_awaited_once_with(
        state,
        context,
    )


@pytest.mark.asyncio
async def test_executor_accepts_runtime_state(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should normalize a RuntimeState result."""

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value=state,
    )

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
    )

    result = await executor.execute(
        state,
        context,
    )

    assert isinstance(
        result,
        RuntimeResult,
    )

    assert result.request_id == state.request_id
    assert result.workflow is Workflow.CHAT
    assert result.output == "Hello from Weings."


@pytest.mark.asyncio
async def test_executor_sets_context_state(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should expose the state through the context."""

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value=make_result(state),
    )

    context.state = None

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
    )

    await executor.execute(
        state,
        context,
    )

    assert context.state is state


@pytest.mark.asyncio
async def test_executor_records_success_metrics(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should record successful executions."""

    metrics = RuntimeMetrics()

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value=make_result(state),
    )

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
        metrics=metrics,
    )

    await executor.execute(
        state,
        context,
    )

    assert metrics.executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    assert metrics.total_tokens == 15


@pytest.mark.asyncio
async def test_executor_wraps_workflow_failure(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should wrap workflow failures."""

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        side_effect=ValueError("provider failed"),
    )

    metrics = RuntimeMetrics()

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
        metrics=metrics,
    )

    with pytest.raises(
        RuntimeExecutionError,
        match="provider failed",
    ):
        await executor.execute(
            state,
            context,
        )

    assert state.error == "provider failed"
    assert metrics.executions == 1
    assert metrics.failed_executions == 1


@pytest.mark.asyncio
async def test_executor_handles_timeout(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should raise a runtime timeout error."""

    async def slow_workflow(
        received_state: RuntimeState,
        received_context: RuntimeContext,
    ) -> RuntimeResult:
        await asyncio.sleep(1)

        return make_result(
            received_state,
        )

    workflow = MagicMock()
    workflow.execute = slow_workflow

    metrics = RuntimeMetrics()

    executor = RuntimeExecutor(
        workflow,
        timeout=0.01,
        metrics=metrics,
    )

    with pytest.raises(
        RuntimeTimeoutError,
        match="exceeded the configured timeout",
    ):
        await executor.execute(
            state,
            context,
        )

    assert state.error == (
        "Runtime execution exceeded the configured timeout."
    )

    assert metrics.executions == 1
    assert metrics.failed_executions == 1


@pytest.mark.asyncio
async def test_executor_handles_cancellation(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should translate task cancellation."""

    async def cancelled_workflow(
        received_state: RuntimeState,
        received_context: RuntimeContext,
    ) -> RuntimeResult:
        raise asyncio.CancelledError()

    workflow = MagicMock()
    workflow.execute = cancelled_workflow

    metrics = RuntimeMetrics()

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
        metrics=metrics,
    )

    with pytest.raises(
        RuntimeCancelledError,
        match="Runtime execution was cancelled",
    ):
        await executor.execute(
            state,
            context,
        )

    assert state.error == (
        "Runtime execution was cancelled."
    )

    assert metrics.executions == 1
    assert metrics.cancelled_executions == 1


@pytest.mark.asyncio
async def test_executor_rejects_invalid_workflow_result(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should reject unsupported workflow results."""

    workflow = MagicMock()
    workflow.execute = AsyncMock(
        return_value="invalid",
    )

    executor = RuntimeExecutor(
        workflow,
        timeout=5,
    )

    with pytest.raises(
        RuntimeExecutionError,
        match="must return RuntimeState or RuntimeResult",
    ):
        await executor.execute(
            state,
            context,
        )


def test_executor_rejects_invalid_timeout() -> None:
    """Executor should reject non-positive timeouts."""

    workflow = MagicMock()

    with pytest.raises(
        ValueError,
        match="timeout must be greater than zero",
    ):
        RuntimeExecutor(
            workflow,
            timeout=0,
        )


def test_executor_exposes_dependencies() -> None:
    """Executor should expose its configured dependencies."""

    workflow = MagicMock()
    metrics = RuntimeMetrics()

    executor = RuntimeExecutor(
        workflow,
        timeout=12,
        metrics=metrics,
    )

    assert executor.workflow is workflow
    assert executor.timeout == 12
    assert executor.metrics is metrics
    