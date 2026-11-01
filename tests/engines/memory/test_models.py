"""Tests for memory database models."""

from uuid import uuid4

import pytest
from sqlalchemy import inspect

from app.engines.memory.models import Memory
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.mark.unit
def test_memory_table_name() -> None:
    """Memory should use the expected database table."""

    assert Memory.__tablename__ == "memories"


@pytest.mark.unit
def test_memory_has_primary_key() -> None:
    """Memory should define a primary key."""

    mapper = inspect(Memory)

    primary_keys = mapper.primary_key

    assert len(primary_keys) == 1
    assert primary_keys[0].name == "id"


@pytest.mark.unit
def test_memory_has_required_columns() -> None:
    """Memory should expose all required persistence columns."""

    columns = {
        column.name
        for column in Memory.__table__.columns
    }

    assert {
        "id",
        "content",
        "memory_type",
        "source",
        "status",
        "importance",
        "metadata",
        "created_at",
        "updated_at",
    }.issubset(columns)


@pytest.mark.unit
def test_memory_column_nullability() -> None:
    """Core memory fields should be non-nullable."""

    table = Memory.__table__

    assert table.c.content.nullable is False
    assert table.c.memory_type.nullable is False
    assert table.c.source.nullable is False
    assert table.c.status.nullable is False
    assert table.c.importance.nullable is False
    assert table.c.metadata.nullable is False


@pytest.mark.unit
def test_memory_defaults() -> None:
    """Memory should define sensible persistence defaults."""

    table = Memory.__table__

    assert table.c.source.default is not None
    assert table.c.status.default is not None
    assert table.c.importance.default is not None
    assert table.c.metadata.default is not None


@pytest.mark.unit
def test_memory_can_be_constructed() -> None:
    """Memory should accept valid model data."""

    memory = Memory(
        id=uuid4(),
        content="User likes AI.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.8,
        metadata_={"topic": "AI"},
    )

    assert memory.content == "User likes AI."
    assert memory.memory_type is MemoryType.PREFERENCE
    assert memory.source is MemorySource.CONVERSATION
    assert memory.status is MemoryStatus.ACTIVE
    assert memory.importance == 0.8
    assert memory.metadata_ == {"topic": "AI"}