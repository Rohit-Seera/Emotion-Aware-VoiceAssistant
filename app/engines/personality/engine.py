"""Personality engine."""

from __future__ import annotations

from dataclasses import dataclass

from app.runtime.state import RuntimeState


@dataclass(slots=True, frozen=True)
class PersonalityProfile:
    """Describe the conversational personality."""

    name: str = "Aiko"

    tone: str = "warm"

    traits: tuple[str, ...] = (
        "friendly",
        "empathetic",
        "supportive",
    )

    response_style: str = (
        "natural, conversational, and concise"
    )


class PersonalityEngine:
    """Provide stable personality configuration."""

    def __init__(
        self,
        profile: PersonalityProfile | None = None,
    ) -> None:
        self._profile = (
            profile
            or PersonalityProfile()
        )

    @property
    def profile(self) -> PersonalityProfile:
        """Return the active personality profile."""

        return self._profile

    def get_traits(self) -> tuple[str, ...]:
        """Return configured personality traits."""

        return self._profile.traits

    def build_instructions(self) -> str:
        """Build personality instructions for the prompt layer."""

        traits = ", ".join(
            self._profile.traits,
        )

        return (
            f"You are {self._profile.name}. "
            f"Your tone is {self._profile.tone}. "
            f"Your personality traits are: {traits}. "
            f"Your response style should be "
            f"{self._profile.response_style}."
        )

    async def get_personality(
        self,
        state: RuntimeState,
    ) -> dict[str, object]:
        """Return runtime-ready personality context.

        The ``state`` argument keeps this engine compatible with the runtime
        node protocol while allowing future per-character profiles.
        """

        del state

        return {
            "name": self._profile.name,
            "tone": self._profile.tone,
            "traits": list(self._profile.traits),
            "response_style": self._profile.response_style,
            "instructions": self.build_instructions(),
        }
