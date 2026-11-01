"""Adapter for runtime workflows."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.runtime.context import RuntimeContext
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


class RuntimeWorkflowAdapter:
    """Adapt a workflow implementation to the runtime interface."""

    def __init__(self, workflow: Any) -> None:
        self._workflow = workflow

    @property
    def workflow(self) -> Any:
        """Return the wrapped workflow."""

        return self._workflow

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Execute the wrapped workflow."""

        result = await self._workflow.execute(
            state,
            context,
        )

        if isinstance(result, RuntimeResult):
            return result

        raise TypeError(
            "Workflow execute() must return RuntimeResult."
        )

    async def stream(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> AsyncIterator[object]:
        """Stream results from the wrapped workflow."""

        stream_method = getattr(
            self._workflow,
            "stream",
            None,
        )

        if stream_method is None:
            raise TypeError(
                "Workflow does not support streaming."
            )

        async for event in stream_method(
            state,
            context,
        ):
            yield event