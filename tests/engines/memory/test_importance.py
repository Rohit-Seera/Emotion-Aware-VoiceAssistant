"""Tests for memory importance scoring."""

import pytest

from app.engines.memory.importance import MemoryImportance
from app.engines.memory.types import MemoryType


@pytest.fixture
def importance() -> MemoryImportance:
    """Return an importance calculator."""

    return MemoryImportance()


def test_empty_content_returns_zero(
    importance: MemoryImportance,
) -> None:
    """Empty content should have zero importance."""

    assert (
        importance.calculate(
            "",
            memory_type=MemoryType.PREFERENCE,
        )
        == 0.0
    )


def test_identity_is_more_important_than_preference(
    importance: MemoryImportance,
) -> None:
    """Identity memories should receive a higher base score."""

    identity = importance.calculate(
        "User's name is Rohit.",
        memory_type=MemoryType.IDENTITY,
    )

    preference = importance.calculate(
        "User likes Python.",
        memory_type=MemoryType.PREFERENCE,
    )

    assert identity > preference


def test_score_is_normalized(
    importance: MemoryImportance,
) -> None:
    """Importance must remain between zero and one."""

    score = importance.calculate(
        "A " * 100,
        memory_type=MemoryType.GOAL,
    )

    assert 0.0 <= score <= 1.0


def test_longer_content_gets_small_bonus(
    importance: MemoryImportance,
) -> None:
    """Meaningful longer content should receive a small bonus."""

    short = importance.calculate(
        "User likes AI.",
        memory_type=MemoryType.PREFERENCE,
    )

    long = importance.calculate(
        "User likes AI and wants to build AI agents "
        "and eventually create a startup around AI products.",
        memory_type=MemoryType.PREFERENCE,
    )

    assert long > short