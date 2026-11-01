"""Tests for memory engine types."""

import pytest

from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.mark.unit
def test_memory_type_values() -> None:
    """MemoryType should expose all supported memory categories."""

    assert MemoryType.IDENTITY.value == "identity"
    assert MemoryType.PREFERENCE.value == "preference"
    assert MemoryType.GOAL.value == "goal"
    assert MemoryType.EMOTIONAL_EVENT.value == "emotional_event"
    assert MemoryType.RELATIONSHIP_EVENT.value == "relationship_event"
    assert (
        MemoryType.CONVERSATION_SUMMARY.value
        == "conversation_summary"
    )


@pytest.mark.unit
def test_memory_status_values() -> None:
    """MemoryStatus should expose supported lifecycle states."""

    assert MemoryStatus.ACTIVE.value == "active"
    assert MemoryStatus.ARCHIVED.value == "archived"


@pytest.mark.unit
def test_memory_source_values() -> None:
    """MemorySource should expose supported memory origins."""

    assert MemorySource.CONVERSATION.value == "conversation"
    assert MemorySource.USER.value == "user"
    assert MemorySource.SYSTEM.value == "system"


@pytest.mark.unit
def test_memory_types_are_strings() -> None:
    """Memory enums should behave like strings."""

    assert isinstance(MemoryType.IDENTITY, str)
    assert isinstance(MemoryStatus.ACTIVE, str)
    assert isinstance(MemorySource.CONVERSATION, str)