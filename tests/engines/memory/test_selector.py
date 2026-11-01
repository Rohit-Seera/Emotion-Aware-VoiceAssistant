"""Tests for memory result selection."""

from uuid import uuid4

import pytest

from app.engines.memory.schemas import (
    MemoryRead,
    MemorySearchResult,
)
from app.engines.memory.selector import MemorySelector
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.fixture
def selector() -> MemorySelector:
    """Return a memory selector."""

    return MemorySelector()


def make_result(
    score: float,
) -> MemorySearchResult:
    """Create a minimal search result."""

    memory = MemoryRead(
        id=uuid4(),
        content="Test memory",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.5,
        metadata={},
        created_at="2026-08-09T00:00:00Z",
        updated_at="2026-08-09T00:00:00Z",
    )

    return MemorySearchResult(
        memory=memory,
        score=score,
    )


def test_select_returns_top_k(
    selector: MemorySelector,
) -> None:
    """Selector should return only the requested number of memories."""

    memories = [
        make_result(0.9),
        make_result(0.8),
        make_result(0.7),
        make_result(0.6),
        make_result(0.5),
    ]

    result = selector.select(
        memories,
        top_k=3,
    )

    assert len(result) == 3
    assert result == memories[:3]


def test_select_returns_all_when_fewer_than_top_k(
    selector: MemorySelector,
) -> None:
    """Selector should return all memories when fewer are available."""

    memories = [
        make_result(0.9),
        make_result(0.8),
    ]

    result = selector.select(
        memories,
        top_k=5,
    )

    assert len(result) == 2
    assert result == memories


def test_select_empty_sequence(
    selector: MemorySelector,
) -> None:
    """Selecting from an empty sequence should return an empty list."""

    result = selector.select(
        [],
        top_k=5,
    )

    assert result == []


def test_select_non_positive_top_k(
    selector: MemorySelector,
) -> None:
    """Non-positive top_k should return no memories."""

    memories = [
        make_result(0.9),
        make_result(0.8),
    ]

    assert selector.select(
        memories,
        top_k=0,
    ) == []

    assert selector.select(
        memories,
        top_k=-1,
    ) == []