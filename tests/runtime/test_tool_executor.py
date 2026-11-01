"""Tests for the tool-executor runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.tool_executor import ToolExecutorNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="What's the weather?",
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
            "tool_route": {
                "name": "weather",
                "arguments": {
                    "city": "Jaipur",
                },
            },
        },
    )


def make_tool() -> MagicMock:
    """Create a mocked runtime tool."""

    tool = MagicMock()
    tool.execute = AsyncMock(
        return_value={
            "temperature": 31,
            "condition": "clear",
        },
    )

    return tool


def make_registry(
    tool: MagicMock,
) -> MagicMock:
    """Create a mocked tool registry."""

    registry = MagicMock()
    registry.get_tool.return_value = tool

    return registry


@pytest.mark.asyncio
async def test_executes_routed_tool(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should execute the routed tool."""

    tool = make_tool()
    registry = make_registry(tool)

    node = ToolExecutorNode(registry)

    result = await node.execute(
        state,
        context,
    )

    assert result is state

    tool.execute.assert_awaited_once_with(
        {
            "city": "Jaipur",
        },
    )


@pytest.mark.asyncio
async def test_resolves_correct_tool(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should resolve the requested tool."""

    tool = make_tool()
    registry = make_registry(tool)

    node = ToolExecutorNode(registry)

    await node.execute(
        state,
        context,
    )

    registry.get_tool.assert_called_once_with(
        "weather",
    )


@pytest.mark.asyncio
async def test_stores_tool_result(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should store the tool result."""

    tool = make_tool()
    registry = make_registry(tool)

    node = ToolExecutorNode(registry)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["tool_result"] == {
        "temperature": 31,
        "condition": "clear",
    }


@pytest.mark.asyncio
async def test_no_tool_route_does_nothing(
    state: RuntimeState,
) -> None:
    """Executor should do nothing without a routed tool."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )

    registry = MagicMock()

    node = ToolExecutorNode(registry)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["tool_result"] is None

    registry.get_tool.assert_not_called()


@pytest.mark.asyncio
async def test_uses_context_registry(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should use the registry from runtime context."""

    tool = make_tool()
    registry = make_registry(tool)

    context.services["tool_registry"] = registry

    node = ToolExecutorNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["tool_result"] == {
        "temperature": 31,
        "condition": "clear",
    }

    registry.get_tool.assert_called_once_with(
        "weather",
    )


@pytest.mark.asyncio
async def test_missing_registry_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Executor should fail without a configured registry."""

    node = ToolExecutorNode()

    with pytest.raises(
        RuntimeError,
        match="Tool registry has not been configured",
    ):
        await node.execute(
            state,
            context,
        )


@pytest.mark.asyncio
async def test_invalid_route_raises(
    state: RuntimeState,
) -> None:
    """Executor should reject invalid tool routes."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "tool_route": "weather",
        },
    )

    registry = MagicMock()

    node = ToolExecutorNode(registry)

    with pytest.raises(
        ValueError,
        match="tool route must be a dictionary",
    ):
        await node.execute(
            state,
            context,
        )

    registry.get_tool.assert_not_called()