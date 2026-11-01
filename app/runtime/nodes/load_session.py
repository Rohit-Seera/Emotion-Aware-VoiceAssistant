"""Load persisted session data into runtime state."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class SessionLoaderProtocol(Protocol):
    """Contract for loading a persisted session."""

    async def load(
        self,
        session_id: UUID,
    ) -> dict[str, Any] | None:
        """Load session data by identifier."""
        ...


class LoadSessionNode:
    """Load session data for the current runtime execution."""

    def __init__(
        self,
        loader: SessionLoaderProtocol | None = None,
    ) -> None:
        self._loader = loader

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Load session data into the runtime state."""

        if state.session_id is None:
            return state

        loader = self._loader

        if loader is None:
            if not context.has_service("session_loader"):
                raise RuntimeError(
                    "Session loader has not been configured."
                )

            loader = context.get_service(
                "session_loader",
            )

        session = await loader.load(
            state.session_id,
        )

        if session is None:
            return state

        state.metadata["session"] = session

        return state