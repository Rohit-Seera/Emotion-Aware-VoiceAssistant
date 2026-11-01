"""Tests for memory result ranking."""

from uuid import uuid4

import pytest

from app.engines.memory.ranker import MemoryRanker
from app.engines.memory.schemas import (
    MemoryRead,
    MemorySearchResult,
)
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.fixture
def ranker() -> MemoryRanker:
    """Return a memory ranker."""

    return MemoryRanker()


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


def test_rank_orders_highest_score_first(
    ranker: MemoryRanker,
) -> None:
    """Highest-scoring memories should come first."""

    low = make_result(0.2)
    high = make_result(0.9)
    medium = make_result(0.5)

    result = ranker.rank(
        [low, high, medium],
    )

    assert result == [high, medium, low]

def test_rank_preserves_all_results(
    ranker: MemoryRanker,
) -> None:
    """Ranking should not remove any memories."""

    memories = [
        make_result(0.2),
        make_result(0.8),
        make_result(0.4),
    ]

    result = ranker.rank(memories)

    assert len(result) == len(memories)

    assert all(
        memory in result
        for memory in memories
    )
def test_rank_empty_sequence(
    ranker: MemoryRanker,
) -> None:
    """Ranking an empty sequence should return an empty list."""

    assert ranker.rank([]) == []


def test_rank_does_not_mutate_input(
    ranker: MemoryRanker,
) -> None:
    """Ranking should not modify the original sequence."""

    low = make_result(0.2)
    high = make_result(0.9)

    memories = [low, high]

    ranker.rank(memories)

    assert memories == [low, high]