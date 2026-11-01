"""Tests for the top-level runtime facade."""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.result import RuntimeResult
from app.runtime.runtime import Runtime
from app.runtime.state import RuntimeState


def make_result(
    request_id: UUID,
    workflow: Workflow,
) -> RuntimeResult:
    """Create a sample runtime result."""

    return RuntimeResult(
        request_id=request_id,
        workflow=workflow,
        status=RuntimeStatus.COMPLETED,
        output="Hello from Weings.",
    )


@pytest.mark.asyncio
async def test_execute_delegates_to_executor() -> None:
    """Runtime should delegate execution to the executor."""

    request_id = uuid4()

    executor = MagicMock()
    executor.execute = AsyncMock(
        return_value=make_result(
            request_id,
            Workflow.CHAT,
        ),
    )

    runtime = Runtime(executor)

    result = await runtime.execute(
        workflow=Workflow.CHAT,
        input="Hello.",
        request_id=request_id,
    )

    assert result.output == (
        "Hello from Weings."
    )

    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_creates_runtime_state() -> None:
    """Runtime should construct the correct state."""

    request_id = uuid4()

    executor = MagicMock()

    async def execute(
        state: RuntimeState,
        context,
    ) -> RuntimeResult:
        assert state.request_id == request_id
        assert state.workflow is Workflow.VOICE
        assert state.input == "Hello Weings."
        assert state.session_id is None

        return make_result(
            request_id,
            Workflow.VOICE,
        )

    executor.execute = execute

    runtime = Runtime(executor)

    result = await runtime.execute(
        workflow=Workflow.VOICE,
        input="Hello Weings.",
        request_id=request_id,
    )

    assert result.workflow is Workflow.VOICE


@pytest.mark.asyncio
async def test_execute_passes_session_id() -> None:
    """Runtime should pass the session identifier into state."""

    request_id = uuid4()
    session_id = uuid4()

    executor = MagicMock()

    async def execute(
        state: RuntimeState,
        context,
    ) -> RuntimeResult:
        assert state.session_id == session_id

        return make_result(
            request_id,
            Workflow.CHAT,
        )

    executor.execute = execute

    runtime = Runtime(executor)

    await runtime.execute(
        workflow=Workflow.CHAT,
        input="Hello.",
        request_id=request_id,
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_execute_passes_services_and_metadata() -> None:
    """Runtime should pass services and metadata into context."""

    request_id = uuid4()

    service = object()

    executor = MagicMock()

    async def execute(
        state: RuntimeState,
        context,
    ) -> RuntimeResult:
        assert context.services["memory_engine"] is service
        assert context.metadata["source"] == "voice"

        return make_result(
            request_id,
            Workflow.VOICE,
        )

    executor.execute = execute

    runtime = Runtime(executor)

    await runtime.execute(
        workflow=Workflow.VOICE,
        input="Hello.",
        request_id=request_id,
        services={
            "memory_engine": service,
        },
        metadata={
            "source": "voice",
        },
    )


@pytest.mark.asyncio
async def test_generates_request_id_when_missing() -> None:
    """Runtime should generate a request ID when none is supplied."""

    executor = MagicMock()

    captured_request_id: UUID | None = None

    async def execute(
        state: RuntimeState,
        context,
    ) -> RuntimeResult:
        nonlocal captured_request_id

        captured_request_id = state.request_id

        return make_result(
            state.request_id,
            Workflow.CHAT,
        )

    executor.execute = execute

    runtime = Runtime(executor)

    result = await runtime.execute(
        workflow=Workflow.CHAT,
        input="Hello.",
    )

    assert captured_request_id is not None
    assert isinstance(
        captured_request_id,
        UUID,
    )
    assert result.request_id == captured_request_id


@pytest.mark.asyncio
async def test_rejects_empty_input() -> None:
    """Runtime should reject empty input."""

    executor = MagicMock()
    executor.execute = AsyncMock()

    runtime = Runtime(executor)

    with pytest.raises(
        ValueError,
        match="Runtime input cannot be empty",
    ):
        await runtime.execute(
            workflow=Workflow.CHAT,
            input="   ",
        )

    executor.execute.assert_not_awaited()


def test_exposes_executor() -> None:
    """Runtime should expose its configured executor."""

    executor = MagicMock()

    runtime = Runtime(executor)

    assert runtime.executor is executor