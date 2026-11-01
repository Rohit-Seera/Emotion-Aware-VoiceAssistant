"""Top-level runtime facade."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol
from uuid import UUID, uuid4

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.executor import RuntimeExecutor
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState


class RuntimeExecutorProtocol(Protocol):
    """Contract for runtime execution."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Execute runtime state."""
        ...


class Runtime:
    """Public entry point for runtime execution."""

    def __init__(
        self,
        executor: RuntimeExecutorProtocol,
    ) -> None:
        self._executor = executor

    @property
    def executor(self) -> RuntimeExecutorProtocol:
        """Return the configured executor."""

        return self._executor

    async def execute(
        self,
        *,
        workflow: Workflow,
        input: str,
        request_id: UUID | None = None,
        session_id: UUID | None = None,
        metadata: dict[str, object] | None = None,
        services: dict[str, object] | None = None,
    ) -> RuntimeResult:
        """Execute a runtime workflow."""

        if not input.strip():
            raise ValueError(
                "Runtime input cannot be empty."
            )

        resolved_request_id = (
            request_id
            or uuid4()
        )

        state = RuntimeState(
            request_id=resolved_request_id,
            workflow=workflow,
            input=input,
            session_id=session_id,
        )

        context = RuntimeContext(
            request_id=resolved_request_id,
            workflow=workflow,
            state=state,
            services=dict(
                services or {},
            ),
            metadata=dict(
                metadata or {},
            ),
        )

        return await self._executor.execute(
            state,
            context,
        )