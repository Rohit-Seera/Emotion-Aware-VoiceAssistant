"""Load relationship context for the runtime execution."""

from __future__ import annotations

from typing import Any, Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class RelationshipProviderProtocol(Protocol):
    """Contract for providing relationship context."""

    async def get_relationship(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Return relationship context for the current state."""
        ...


class RelationshipNode:
    """Load relationship context into runtime context."""

    def __init__(
        self,
        provider: RelationshipProviderProtocol | None = None,
    ) -> None:
        self._provider = provider

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Load relationship context."""

        provider = self._provider

        if provider is None:
            if not context.has_service("relationship_provider"):
                raise RuntimeError(
                    "Relationship provider has not been configured."
                )

            provider = context.get_service(
                "relationship_provider",
            )

        relationship = await provider.get_relationship(
            state,
        )

        context.metadata["relationship"] = relationship

        return state