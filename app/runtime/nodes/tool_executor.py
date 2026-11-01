"""Execute a routed runtime tool."""

from __future__ import annotations

from typing import Any, Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class RuntimeToolProtocol(Protocol):
    """Contract for an executable runtime tool."""

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute the tool."""
        ...


class ToolRegistryProtocol(Protocol):
    """Contract for resolving runtime tools."""

    def get_tool(
        self,
        name: str,
    ) -> RuntimeToolProtocol:
        """Return a registered tool."""
        ...


class ToolExecutorNode:
    """Execute the tool selected by the tool router."""

    def __init__(
        self,
        registry: ToolRegistryProtocol | None = None,
    ) -> None:
        self._registry = registry

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Execute the routed tool."""

        route = context.metadata.get(
            "tool_route",
        )

        if route is None:
            context.metadata["tool_result"] = None
            return state

        if not isinstance(route, dict):
            raise ValueError(
                "Runtime tool route must be a dictionary."
            )

        name = route.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Runtime tool route must contain a valid name."
            )

        arguments = route.get(
            "arguments",
            {},
        )

        if not isinstance(arguments, dict):
            raise ValueError(
                "Runtime tool route arguments must be a dictionary."
            )

        registry = self._registry

        if registry is None:
            if not context.has_service("tool_registry"):
                raise RuntimeError(
                    "Tool registry has not been configured."
                )

            registry = context.get_service(
                "tool_registry",
            )

        tool = registry.get_tool(name)

        result = await tool.execute(
            dict(arguments),
        )

        context.metadata["tool_result"] = result

        return state