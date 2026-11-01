"""Persist a runtime memory through the memory engine."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from app.engines.memory.schemas import MemoryCreate, MemoryRead
from app.engines.memory.types import MemorySource, MemoryType
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class MemoryEngineProtocol(Protocol):
    """Contract for the memory engine used by the node."""

    async def create(self, memory: MemoryCreate) -> MemoryRead:
        """Create and persist a memory."""
        ...


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


class SaveMemoryNode:
    """Save a memory generated during runtime execution."""

    def __init__(
        self,
        memory_engine: MemoryEngineProtocol | None = None,
    ) -> None:
        self._memory_engine = memory_engine

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Persist the configured memory."""

        memory_data = context.metadata.get("memory_to_save")
        if memory_data is None:
            context.metadata["saved_memory"] = None
            return state

        if not isinstance(memory_data, dict):
            raise ValueError("Memory to save must be a dictionary.")

        content = memory_data.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Memory content must be a non-empty string."
            )

        memory_type = memory_data.get(
            "memory_type", MemoryType.CONVERSATION_SUMMARY
        )
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError as error:
                raise ValueError(
                    f"Invalid memory type: {memory_type}"
                ) from error
        if not isinstance(memory_type, MemoryType):
            raise ValueError("Memory type must be a valid MemoryType.")

        source = memory_data.get("source", MemorySource.CONVERSATION)
        if isinstance(source, str):
            try:
                source = MemorySource(source)
            except ValueError as error:
                raise ValueError(
                    f"Invalid memory source: {source}"
                ) from error
        if not isinstance(source, MemorySource):
            raise ValueError("Memory source must be a valid MemorySource.")

        # Resolve the engine before creating the schema so a missing service
        # produces the expected configuration error, not a validation error.
        memory_engine = self._memory_engine
        if memory_engine is None:
            if not context.has_service("memory_engine"):
                raise RuntimeError(
                    "Memory engine has not been configured."
                )
            memory_engine = context.get_service("memory_engine")

        # Scope IDs come from trusted runtime metadata, never model-generated
        # memory content, to prevent cross-user/cross-character memory writes.
        user_id = _memory_scope_id(context, "user_id")
        character_id = _memory_scope_id(context, "character_id")

        memory = MemoryCreate(
            user_id=user_id,
            character_id=character_id,
            content=content,
            memory_type=memory_type,
            source=source,
            importance=memory_data.get("importance", 0.5),
            metadata=memory_data.get("metadata", {}),
        )

        saved_memory = await memory_engine.create(memory)
        context.metadata["saved_memory"] = saved_memory
        return state
