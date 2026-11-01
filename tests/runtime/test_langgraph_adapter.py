"""Tests for the LangGraph runtime adapter."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.adapters.langgraph import LangGraphAdapter
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


def make_state() -> RuntimeState:
    """Create a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        status=RuntimeStatus.COMPLETED,
        output="Hi!",
        metadata={"provider": "openai"},
        token_usage=RuntimeTokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def make_context(
    state: RuntimeState,
) -> RuntimeContext:
    """Create a sample runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )


@pytest.mark.asyncio
async def test_adapter_stores_graph() -> None:
    """Adapter should store the compiled graph."""

    graph = MagicMock()

    adapter = LangGraphAdapter(graph)

    assert adapter.graph is graph


@pytest.mark.asyncio
async def test_invoke_calls_ainvoke() -> None:
    """invoke should call LangGraph ainvoke."""

    state = make_state()
    context = make_context(state)

    graph = MagicMock()

    graph.ainvoke = AsyncMock(
        return_value={
            "request_id": state.request_id,
            "workflow": state.workflow,
            "input": state.input,
            "status": RuntimeStatus.COMPLETED,
            "output": "Done",
            "error": None,
            "memories": [],
            "metadata": {},
            "token_usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
    )

    adapter = LangGraphAdapter(graph)

    result = await adapter.invoke(
        state,
        context,
    )

    assert result.output == "Done"
    assert result.status is RuntimeStatus.COMPLETED

    graph.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_invoke_passes_thread_id() -> None:
    """invoke should pass the request ID as thread ID."""

    state = make_state()
    context = make_context(state)

    graph = MagicMock()

    graph.ainvoke = AsyncMock(
        return_value={
            "request_id": state.request_id,
            "workflow": state.workflow,
            "input": state.input,
            "status": RuntimeStatus.COMPLETED,
            "output": "Done",
            "error": None,
            "memories": [],
            "metadata": {},
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
    )

    adapter = LangGraphAdapter(graph)

    await adapter.invoke(
        state,
        context,
    )

    kwargs = graph.ainvoke.await_args.kwargs

    assert kwargs["config"]["configurable"]["thread_id"] == (
        str(context.request_id)
    )


@pytest.mark.asyncio
async def test_invoke_preserves_runtime_metadata() -> None:
    """invoke should preserve runtime metadata."""

    state = make_state()
    context = make_context(state)

    graph = MagicMock()

    graph.ainvoke = AsyncMock(
        return_value={
            "request_id": state.request_id,
            "workflow": state.workflow,
            "input": state.input,
            "status": RuntimeStatus.COMPLETED,
            "output": "Done",
            "error": None,
            "memories": [],
            "metadata": {
                "model": "test-model",
            },
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 6,
                "total_tokens": 10,
            },
        },
    )

    adapter = LangGraphAdapter(graph)

    result = await adapter.invoke(
        state,
        context,
    )

    assert result.metadata == {
        "model": "test-model",
    }

    assert result.token_usage.total_tokens == 10


@pytest.mark.asyncio
async def test_stream_calls_astream() -> None:
    """stream should delegate to LangGraph astream."""

    state = make_state()
    context = make_context(state)

    graph = MagicMock()

    async def fake_stream(
        graph_state,
        *,
        config,
    ):
        yield {"node": "start"}
        yield {"node": "finish"}

    graph.astream = fake_stream

    adapter = LangGraphAdapter(graph)

    updates = []

    async for update in adapter.stream(
        state,
        context,
    ):
        updates.append(update)

    assert updates == [
        {"node": "start"},
        {"node": "finish"},
    ]


@pytest.mark.asyncio
async def test_stream_passes_thread_id() -> None:
    """stream should pass the request ID as thread ID."""

    state = make_state()
    context = make_context(state)

    graph = MagicMock()
    captured = {}

    async def fake_stream(
        graph_state,
        *,
        config,
    ):
        captured["config"] = config
        yield {"node": "test"}

    graph.astream = fake_stream

    adapter = LangGraphAdapter(graph)

    async for _ in adapter.stream(
        state,
        context,
    ):
        pass

    assert (
        captured["config"]["configurable"]["thread_id"]
        == str(context.request_id)
    )