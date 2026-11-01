"""Finish runtime execution."""

from __future__ import annotations

from app.core.enums import RuntimeStatus
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class FinishNode:
    """Finalize a successful runtime execution."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Mark the runtime execution as completed."""

        state.status = RuntimeStatus.COMPLETED

        if state.output is None:
            state.output = ""

        context.metadata["finished"] = True

        return state