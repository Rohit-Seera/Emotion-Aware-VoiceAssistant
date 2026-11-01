"""Tests for the relationship engine."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.engines.relationship.engine import (
    RelationshipEngine,
    RelationshipProfile,
)
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
    )


@pytest.mark.asyncio
async def test_default_relationship(
    state: RuntimeState,
) -> None:
    """Engine should provide the default relationship context."""

    engine = RelationshipEngine()

    relationship = await engine.get_relationship(
        state,
    )

    assert relationship["level"] == "new"
    assert relationship["traits"] == ()


@pytest.mark.asyncio
async def test_custom_relationship_profile(
    state: RuntimeState,
) -> None:
    """Engine should use a custom relationship profile."""

    profile = RelationshipProfile(
        level="familiar",
        traits=(
            "comfortable",
            "trusting",
        ),
    )

    engine = RelationshipEngine(profile)

    relationship = await engine.get_relationship(
        state,
    )

    assert relationship["level"] == "familiar"
    assert relationship["traits"] == (
        "comfortable",
        "trusting",
    )


@pytest.mark.asyncio
async def test_relationship_metadata_is_included(
    state: RuntimeState,
) -> None:
    """Profile metadata should be included in the context."""

    profile = RelationshipProfile(
        level="close",
        metadata={
            "interaction_count": 25,
            "preferred_name": "Rohit",
        },
    )

    engine = RelationshipEngine(profile)

    relationship = await engine.get_relationship(
        state,
    )

    assert relationship["level"] == "close"
    assert relationship["interaction_count"] == 25
    assert relationship["preferred_name"] == "Rohit"


@pytest.mark.asyncio
async def test_relationship_returns_dict(
    state: RuntimeState,
) -> None:
    """Relationship context should be a dictionary."""

    engine = RelationshipEngine()

    relationship = await engine.get_relationship(
        state,
    )

    assert isinstance(
        relationship,
        dict,
    )


def test_profile_is_exposed() -> None:
    """Engine should expose its configured profile."""

    profile = RelationshipProfile(
        level="familiar",
    )

    engine = RelationshipEngine(profile)

    assert engine.profile is profile


def test_profile_is_immutable() -> None:
    """Relationship profile should be immutable."""

    profile = RelationshipProfile()

    try:
        profile.level = "close"
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "RelationshipProfile should be immutable."
        )