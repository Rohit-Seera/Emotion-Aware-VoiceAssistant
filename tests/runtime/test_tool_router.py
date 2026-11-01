"""Tests for the tool-router runtime node."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.tool_router import (
    ToolRoute,
    ToolRouterNode,
)
from app.runtime.state import RuntimeState


class FakeToolRegistry:
    """Simple test tool registry."""

    def __init__(
        self,
        tools: set[str],
    ) -> None:
        self.tools = tools

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """Return whether a tool exists."""

        return name in self.tools


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


def test_routes_valid_tool() -> None:
    """Router should create a valid tool route."""

    registry = FakeToolRegistry({"weather"})

    node = ToolRouterNode(registry)

    result = node.route(
        {
            "name": "weather",
            "arguments": {
                "city": "Jaipur",
            },
        },
    )

    assert result == ToolRoute(
        name="weather",
        arguments={
            "city": "Jaipur",
        },
    )


def test_rejects_missing_tool_name() -> None:
    """Router should reject requests without a tool name."""

    node = ToolRouterNode()

    with pytest.raises(
        ValueError,
        match="valid name",
    ):
        node.route(
            {
                "arguments": {},
            },
        )


def test_rejects_invalid_arguments() -> None:
    """Router should reject non-dictionary arguments."""

    node = ToolRouterNode()

    with pytest.raises(
        ValueError,
        match="arguments must be a dictionary",
    ):
        node.route(
            {
                "name": "weather",
                "arguments": "Jaipur",
            },
        )


@pytest.mark.asyncio
async def test_execute_without_tool_request(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Execution without a tool request should do nothing."""

    node = ToolRouterNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["tool_route"] is None


@pytest.mark.asyncio
async def test_execute_stores_tool_route(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Execution should store the validated tool route."""

    registry = FakeToolRegistry({"weather"})

    context.metadata["tool_request"] = {
        "name": "weather",
        "arguments": {
            "city": "Jaipur",
        },
    }

    node = ToolRouterNode(registry)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["tool_route"] == {
        "name": "weather",
        "arguments": {
            "city": "Jaipur",
        },
    }