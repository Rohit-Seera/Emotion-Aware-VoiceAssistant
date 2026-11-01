"""Load personality configuration into the runtime context."""

from __future__ import annotations

from typing import Any, Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class PersonalityProviderProtocol(Protocol):
    """Contract for an engine that provides personality instructions."""

    async def get_personality(self, state: RuntimeState) -> dict[str, Any]:
        """Return the personality configuration for a conversation."""
        ...


class PersonalityNode:
    """Resolve and attach the current personality profile."""

    def __init__(self, provider: PersonalityProviderProtocol | None = None) -> None:
        self._provider = provider

    async def execute(self, state: RuntimeState, context: RuntimeContext) -> RuntimeState:
        """Load personality from the configured provider and store it in context."""
        provider = self._provider
        if provider is None:
            if not context.has_service("personality_provider"):
                raise RuntimeError("Personality provider has not been configured.")
            provider = context.get_service("personality_provider")

        personality = await provider.get_personality(state)
        context.metadata["personality"] = personality
        return state
