"""Character registry and active-character engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class CharacterProfile:
    """Describe a Emotion Aware Voice Assistant character."""

    id: str = "aiko"
    name: str = "Aiko"

    description: str = (
        "A fictional AI voice companion designed to feel "
        "natural, warm, intelligent, and present."
    )

    # Character-level configuration is intentionally separate from the
    # provider implementation so voices can be swapped later.
    personality: dict[str, Any] = field(default_factory=dict)
    voice_provider: str | None = None
    voice_id: str | None = None

    appearance: dict[str, Any] = field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


class CharacterEngine:
    """Provide the active character and built-in character registry."""

    _BUILT_IN_PROFILES: tuple[CharacterProfile, ...] = (
        CharacterProfile(
            id="aiko",
            name="Aiko",
            description=(
                "A fictional AI voice companion designed to feel "
                "natural, warm, intelligent, and present."
            ),
        ),
        CharacterProfile(
            id="yuki",
            name="Yuki",
            description="A calm and curious AI companion.",
        ),
        CharacterProfile(
            id="mika",
            name="Mika",
            description="An energetic and confident AI companion.",
        ),
        CharacterProfile(
            id="luna",
            name="Luna",
            description="A calm and reflective AI companion.",
        ),
        CharacterProfile(
            id="kaito",
            name="Kaito",
            description="A confident and analytical AI companion.",
        ),
    )

    def __init__(
        self,
        profile: CharacterProfile | None = None,
    ) -> None:
        self._profiles = {
            item.id: item
            for item in self._BUILT_IN_PROFILES
        }

        if profile is not None:
            self._profiles[profile.id] = profile

        # Aiko is the first active/default character. Character switching is
        # deliberately kept out of this MVP step and can be added later.
        self._active_id = profile.id if profile is not None else "aiko"

    @property
    def profile(self) -> CharacterProfile:
        """Return the active character profile."""

        return self._profiles[self._active_id]

    @property
    def active_id(self) -> str:
        """Return the active character identifier."""

        return self._active_id

    def get_character(
        self,
        character_id: str | None = None,
    ) -> dict[str, Any]:
        """Return a character representation by id or the active character."""

        profile = self.get_profile(character_id)

        return {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "personality": dict(profile.personality),
            "voice_provider": profile.voice_provider,
            "voice_id": profile.voice_id,
            "appearance": dict(profile.appearance),
            **profile.metadata,
        }

    def get_profile(
        self,
        character_id: str | None = None,
    ) -> CharacterProfile:
        """Return a profile by id or the active profile."""

        selected_id = character_id or self._active_id

        try:
            return self._profiles[selected_id]
        except KeyError as error:
            raise ValueError(
                f"Unknown character '{selected_id}'."
            ) from error

    def list_characters(self) -> tuple[CharacterProfile, ...]:
        """Return the currently registered built-in/custom profiles."""

        return tuple(self._profiles.values())

    def build_identity(self) -> str:
        """Build a concise character identity."""

        return (
            f"{self.profile.name}: "
            f"{self.profile.description}"
        )
