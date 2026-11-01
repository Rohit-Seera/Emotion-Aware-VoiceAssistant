"""Runtime workflow orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


class WorkflowAdapterProtocol(Protocol):
    """Contract for the workflow execution adapter."""

    async def invoke(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Invoke the underlying workflow."""
        ...

    async def stream(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> AsyncIterator[object]:
        """Stream workflow execution events."""
        ...


class RuntimeWorkflow:
    """Execute a runtime workflow through an adapter."""

    def __init__(
        self,
        adapter: WorkflowAdapterProtocol,
    ) -> None:
        self._adapter = adapter

    @property
    def adapter(self) -> WorkflowAdapterProtocol:
        """Return the configured workflow adapter."""

        return self._adapter

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Execute the workflow."""

        return await self._adapter.invoke(
            state,
            context,
        )

    async def stream(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> AsyncIterator[object]:
        """Stream workflow execution events."""

        async for event in self._adapter.stream(
            state,
            context,
        ):
            yield event