"""Runtime interfaces."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Protocol

from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


class RuntimeNodeProtocol(Protocol):
    """Contract for a runtime workflow node."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Execute the node against the current state."""
        ...


class RuntimeWorkflowProtocol(Protocol):
    """Contract for executable workflows."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Execute a workflow."""
        ...


class RuntimeExecutorProtocol(Protocol):
    """Contract for runtime execution."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Execute runtime state."""
        ...


class RuntimeStreamProtocol(Protocol):
    """Contract for streaming runtime events/results."""

    async def stream(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> AsyncIterator[object]:
        """Stream runtime execution events."""
        ...


class RuntimeRegistryProtocol(Protocol):
    """Contract for registering runtime components."""

    def register_node(
        self,
        name: str,
        node: RuntimeNodeProtocol,
    ) -> None:
        """Register a runtime node."""
        ...

    def get_node(
        self,
        name: str,
    ) -> RuntimeNodeProtocol:
        """Retrieve a registered node."""
        ...

    def has_node(
        self,
        name: str,
    ) -> bool:
        """Check whether a node is registered."""
        ...