"""Tests for memory engine schemas."""

from datetime import datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryRead,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryUpdate,
)
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.mark.unit
def test_memory_create_defaults() -> None:
    """MemoryCreate should provide sensible defaults."""

    memory = MemoryCreate(
        content="User likes anime.",
        memory_type=MemoryType.PREFERENCE,
    )

    assert memory.content == "User likes anime."
    assert memory.memory_type is MemoryType.PREFERENCE
    assert memory.source is MemorySource.CONVERSATION
    assert memory.importance == 0.5
    assert memory.metadata == {}


@pytest.mark.unit
def test_memory_create_accepts_valid_values() -> None:
    """MemoryCreate should accept valid memory data."""

    memory = MemoryCreate(
        content="User wants to build an AI startup.",
        memory_type=MemoryType.GOAL,
        source=MemorySource.USER,
        importance=0.9,
        metadata={"topic": "startup"},
    )

    assert memory.memory_type is MemoryType.GOAL
    assert memory.source is MemorySource.USER
    assert memory.importance == 0.9
    assert memory.metadata["topic"] == "startup"


@pytest.mark.unit
def test_memory_create_rejects_empty_content() -> None:
    """MemoryCreate should reject empty content."""

    with pytest.raises(ValidationError):
        MemoryCreate(
            content="",
            memory_type=MemoryType.IDENTITY,
        )


@pytest.mark.unit
def test_memory_create_rejects_invalid_importance() -> None:
    """Importance must remain between zero and one."""

    with pytest.raises(ValidationError):
        MemoryCreate(
            content="Important memory.",
            memory_type=MemoryType.IDENTITY,
            importance=1.5,
        )


@pytest.mark.unit
def test_memory_update_allows_partial_updates() -> None:
    """MemoryUpdate should allow fields to be omitted."""

    update = MemoryUpdate(
        importance=0.8,
    )

    assert update.content is None
    assert update.memory_type is None
    assert update.importance == 0.8


@pytest.mark.unit
def test_memory_read_from_attributes() -> None:
    """MemoryRead should support ORM-style attribute objects."""

    now = datetime.now()
    memory_id = uuid4()

    class MemoryObject:
        id = memory_id
        content = "User likes AI."
        memory_type = MemoryType.PREFERENCE
        source = MemorySource.CONVERSATION
        status = MemoryStatus.ACTIVE
        importance = 0.7
        metadata = {}
        created_at = now
        updated_at = now

    memory = MemoryRead.model_validate(MemoryObject())

    assert memory.id == memory_id
    assert memory.content == "User likes AI."
    assert memory.status is MemoryStatus.ACTIVE


@pytest.mark.unit
def test_memory_search_request_defaults() -> None:
    """Search request should use memory retrieval defaults."""

    request = MemorySearchRequest(
        query="What does the user like?",
    )

    assert request.query == "What does the user like?"
    assert request.top_k == 10
    assert request.similarity_threshold == 0.75


@pytest.mark.unit
def test_memory_search_request_rejects_empty_query() -> None:
    """Search requests should require a query."""

    with pytest.raises(ValidationError):
        MemorySearchRequest(query="")


@pytest.mark.unit
def test_memory_search_result() -> None:
    """Search results should contain a memory and similarity score."""

    now = datetime.now()

    memory = MemoryRead(
        id=uuid4(),
        content="User likes AI.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.8,
        metadata={},
        created_at=now,
        updated_at=now,
    )

    result = MemorySearchResult(
        memory=memory,
        score=0.91,
    )

    assert result.memory is memory
    assert result.score == 0.91