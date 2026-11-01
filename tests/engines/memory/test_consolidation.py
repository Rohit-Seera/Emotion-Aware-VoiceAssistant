"""Tests for memory consolidation."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.engines.memory.consolidation import MemoryConsolidator
from app.engines.memory.schemas import MemoryRead
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


def make_memory(
    *,
    content: str,
    importance: float = 0.5,
    memory_type: MemoryType = MemoryType.PREFERENCE,
) -> MemoryRead:
    """Create a test memory."""

    now = datetime.now(timezone.utc)

    return MemoryRead(
        id=uuid4(),
        content=content,
        memory_type=memory_type,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=importance,
        metadata={},
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_empty_memories_return_empty() -> None:
    """Empty input should produce an empty result."""

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate([])

    assert result == []


@pytest.mark.asyncio
async def test_unique_memories_are_preserved() -> None:
    """Unique memories should remain unchanged."""

    first = make_memory(
        content="User likes Python.",
    )

    second = make_memory(
        content="User likes C++.",
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [first, second],
    )

    assert list(result) == [
        first,
        second,
    ]


@pytest.mark.asyncio
async def test_duplicate_memories_are_consolidated() -> None:
    """Duplicate content should produce one memory."""

    first = make_memory(
        content="User likes Python.",
        importance=0.5,
    )

    second = make_memory(
        content="User likes Python.",
        importance=0.5,
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [first, second],
    )

    assert len(result) == 1
    assert result[0] is first


@pytest.mark.asyncio
async def test_highest_importance_duplicate_is_retained() -> None:
    """The strongest duplicate should be retained."""

    weaker = make_memory(
        content="User likes Python.",
        importance=0.4,
    )

    stronger = make_memory(
        content="User likes Python.",
        importance=0.9,
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [weaker, stronger],
    )

    assert len(result) == 1
    assert result[0] is stronger


@pytest.mark.asyncio
async def test_duplicate_matching_is_case_insensitive() -> None:
    """Duplicate matching should ignore letter casing."""

    first = make_memory(
        content="User Likes Python.",
        importance=0.5,
    )

    second = make_memory(
        content="user likes python.",
        importance=0.8,
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [first, second],
    )

    assert len(result) == 1
    assert result[0] is second


@pytest.mark.asyncio
async def test_duplicate_matching_ignores_outer_whitespace() -> None:
    """Duplicate matching should ignore surrounding whitespace."""

    first = make_memory(
        content="User likes Python.",
        importance=0.5,
    )

    second = make_memory(
        content="  User likes Python.  ",
        importance=0.7,
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [first, second],
    )

    assert len(result) == 1
    assert result[0] is second


@pytest.mark.asyncio
async def test_different_memory_types_are_not_merged() -> None:
    """Same content in different categories should remain separate."""

    preference = make_memory(
        content="User likes Python.",
        memory_type=MemoryType.PREFERENCE,
        importance=0.5,
    )

    goal = make_memory(
        content="User likes Python.",
        memory_type=MemoryType.GOAL,
        importance=0.9,
    )

    consolidator = MemoryConsolidator()

    result = await consolidator.consolidate(
        [preference, goal],
    )

    assert len(result) == 2
    assert preference in result
    assert goal in result