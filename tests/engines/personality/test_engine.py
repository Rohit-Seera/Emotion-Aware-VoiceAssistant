"""Tests for the personality engine."""

from app.engines.personality.engine import (
    PersonalityEngine,
    PersonalityProfile,
)


def test_default_profile_is_weings() -> None:
    """Engine should provide the default Weings profile."""

    engine = PersonalityEngine()

    assert engine.profile.name == "Weings"
    assert engine.profile.tone == "warm"


def test_default_traits_are_available() -> None:
    """Default personality traits should be available."""

    engine = PersonalityEngine()

    assert engine.get_traits() == (
        "friendly",
        "empathetic",
        "supportive",
    )


def test_custom_profile_is_supported() -> None:
    """Engine should accept a custom personality profile."""

    profile = PersonalityProfile(
        name="Weings",
        tone="playful",
        traits=(
            "curious",
            "energetic",
        ),
        response_style="casual and expressive",
    )

    engine = PersonalityEngine(profile)

    assert engine.profile is profile
    assert engine.profile.tone == "playful"
    assert engine.get_traits() == (
        "curious",
        "energetic",
    )


def test_build_instructions_contains_identity() -> None:
    """Instructions should contain the personality identity."""

    engine = PersonalityEngine()

    instructions = engine.build_instructions()

    assert "You are Weings." in instructions


def test_build_instructions_contains_tone() -> None:
    """Instructions should contain the configured tone."""

    engine = PersonalityEngine()

    instructions = engine.build_instructions()

    assert "Your tone is warm." in instructions


def test_build_instructions_contains_traits() -> None:
    """Instructions should contain personality traits."""

    engine = PersonalityEngine()

    instructions = engine.build_instructions()

    assert "friendly, empathetic, supportive" in instructions


def test_build_instructions_contains_response_style() -> None:
    """Instructions should contain the response style."""

    engine = PersonalityEngine()

    instructions = engine.build_instructions()

    assert (
        "natural, conversational, and concise"
        in instructions
    )


def test_profile_is_immutable() -> None:
    """Personality profile should be immutable."""

    profile = PersonalityProfile()

    try:
        profile.name = "Other"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "PersonalityProfile should be immutable."
        )