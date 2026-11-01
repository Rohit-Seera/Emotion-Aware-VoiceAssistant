"""Route tool requests for runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class ToolRegistryProtocol(Protocol):
    """Contract for the runtime tool registry."""

    def has_tool(
        self,
        name: str,
    ) -> bool:
        """Return whether a tool is registered."""
        ...


@dataclass(frozen=True, slots=True)
class ToolRoute:
    """A validated tool execution request."""

    name: str
    arguments: dict[str, Any]


class ToolRouterNode:
    """Validate and route requested tools."""

    def __init__(
        self,
        registry: ToolRegistryProtocol | None = None,
    ) -> None:
        self._registry = registry

    def route(
        self,
        tool_request: dict[str, Any],
    ) -> ToolRoute:
        """Validate a tool request."""

        name = tool_request.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Tool request must contain a valid name."
            )

        arguments = tool_request.get(
            "arguments",
            {},
        )

        if not isinstance(arguments, dict):
            raise ValueError(
                "Tool request arguments must be a dictionary."
            )

        registry = self._registry

        if registry is not None and not registry.has_tool(name):
            raise ValueError(
                f"Tool '{name}' is not registered."
            )

        return ToolRoute(
            name=name,
            arguments=dict(arguments),
        )

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Route the current tool request if one exists."""

        tool_request = context.metadata.get(
            "tool_request",
        )

        if tool_request is None:
            context.metadata["tool_route"] = None
            return state

        if not isinstance(tool_request, dict):
            raise ValueError(
                "Runtime tool request must be a dictionary."
            )

        route = self.route(tool_request)

        context.metadata["tool_route"] = {
            "name": route.name,
            "arguments": route.arguments,
        }

        return state