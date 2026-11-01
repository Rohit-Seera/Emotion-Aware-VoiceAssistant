"""Relationship engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.runtime.state import RuntimeState


@dataclass(slots=True, frozen=True)
class RelationshipProfile:
    """Describe the current relationship context."""

    level: str = "new"

    traits: tuple[str, ...] = field(
        default_factory=tuple,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


class RelationshipEngine:
    """Provide relationship context for runtime execution."""

    def __init__(
        self,
        profile: RelationshipProfile | None = None,
    ) -> None:
        self._profile = (
            profile
            or RelationshipProfile()
        )

    @property
    def profile(self) -> RelationshipProfile:
        """Return the active relationship profile."""

        return self._profile

    async def get_relationship(
        self,
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Return relationship context for the current runtime state."""

        return {
            "level": self._profile.level,
            "traits": self._profile.traits,
            **self._profile.metadata,
        }