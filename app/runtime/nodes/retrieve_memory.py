"""Retrieve relevant memories for the current runtime execution."""

from __future__ import annotations

from uuid import UUID

from app.engines.memory.engine import MemoryEngine
from app.engines.memory.schemas import MemorySearchRequest
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


def _memory_scope_id(context: RuntimeContext, key: str) -> UUID:
    """Read a trusted user/character UUID from runtime metadata."""

    raw_value = context.metadata.get(key)
    if raw_value is None:
        raise RuntimeError(
            f"Memory scope '{key}' has not been configured."
        )

    try:
        return UUID(str(raw_value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Memory scope '{key}' must be a valid UUID."
        ) from error


class RetrieveMemoryNode:
    """Retrieve relevant memories for the current runtime execution."""

    def __init__(
        self,
        memory_engine: MemoryEngine | None = None,
    ) -> None:
        self._memory_engine = memory_engine

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Retrieve memories relevant to the current input."""

        memory_engine = self._memory_engine

        if memory_engine is None:
            if not context.has_service("memory_engine"):
                raise RuntimeError(
                    "Memory engine has not been configured."
                )

            memory_engine = context.get_service("memory_engine")

        user_id = _memory_scope_id(context, "user_id")
        character_id = _memory_scope_id(context, "character_id")

        request = MemorySearchRequest(
            user_id=user_id,
            character_id=character_id,
            query=state.input,
        )

        memories = await memory_engine.search(request)
        state.memories = memories
        return state
