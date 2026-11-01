"""Tests for the character engine."""

from app.engines.character.engine import (
    CharacterEngine,
    CharacterProfile,
)


def test_default_character_is_weings() -> None:
    """Engine should provide the default Weings character."""

    engine = CharacterEngine()

    character = engine.get_character()

    assert character["name"] == "Weings"


def test_default_description_is_available() -> None:
    """Default character description should be available."""

    engine = CharacterEngine()

    character = engine.get_character()

    assert (
        "AI voice companion"
        in character["description"]
    )


def test_custom_profile_is_supported() -> None:
    """Engine should support a custom character profile."""

    profile = CharacterProfile(
        name="Weings",
        description="A friendly digital companion.",
    )

    engine = CharacterEngine(profile)

    assert engine.profile is profile

    character = engine.get_character()

    assert character["name"] == "Weings"
    assert (
        character["description"]
        == "A friendly digital companion."
    )


def test_appearance_is_preserved() -> None:
    """Character appearance should be preserved."""

    profile = CharacterProfile(
        appearance={
            "style": "anime",
            "avatar": "live2d",
        },
    )

    engine = CharacterEngine(profile)

    character = engine.get_character()

    assert character["appearance"] == {
        "style": "anime",
        "avatar": "live2d",
    }


def test_metadata_is_included() -> None:
    """Character metadata should be included."""

    profile = CharacterProfile(
        metadata={
            "version": "1.0",
            "mode": "companion",
        },
    )

    engine = CharacterEngine(profile)

    character = engine.get_character()

    assert character["version"] == "1.0"
    assert character["mode"] == "companion"


def test_build_identity() -> None:
    """Engine should build a concise identity string."""

    engine = CharacterEngine()

    identity = engine.build_identity()

    assert identity.startswith(
        "Weings: "
    )

    assert (
        "AI voice companion"
        in identity
    )


def test_profile_is_immutable() -> None:
    """Character profile should be immutable."""

    profile = CharacterProfile()

    try:
        profile.name = "Other"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "CharacterProfile should be immutable."
        )