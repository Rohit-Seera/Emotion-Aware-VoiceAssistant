"""Tests for the memory repository."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.engines.memory.models import Memory
from app.engines.memory.repository import MemoryRepository
from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryUpdate,
)
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.fixture
def session() -> MagicMock:
    """Return a mocked async database session."""

    session = MagicMock()
    session.get = AsyncMock()
    session.flush = AsyncMock()
    session.delete = AsyncMock()
    session.execute = AsyncMock()

    return session


@pytest.fixture
def repository(
    session: MagicMock,
) -> MemoryRepository:
    """Return a memory repository."""

    return MemoryRepository(session)


@pytest.mark.asyncio
async def test_repository_uses_memory_model(
    repository: MemoryRepository,
) -> None:
    """Repository should use the Memory model."""

    assert repository.model is Memory


@pytest.mark.asyncio
async def test_create(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Create should add a memory and flush the session."""

    data = MemoryCreate(
        content="User likes AI.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        importance=0.8,
        metadata={"topic": "AI"},
    )

    result = await repository.create(data)

    assert result.content == "User likes AI."
    assert result.memory_type is MemoryType.PREFERENCE
    assert result.source is MemorySource.CONVERSATION
    assert result.importance == 0.8
    assert result.metadata == {"topic": "AI"}

    session.add.assert_called_once()
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_returns_memory(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Get should return a serialized memory."""

    memory = Memory(
        id=uuid4(),
        content="User likes Python.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.7,
        metadata_={},
    )

    session.get.return_value = memory

    result = await repository.get(memory.id)

    assert result is not None
    assert result.id == memory.id
    assert result.content == "User likes Python."

    session.get.assert_awaited_once_with(
        Memory,
        memory.id,
    )


@pytest.mark.asyncio
async def test_get_returns_none(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Get should return None when memory does not exist."""

    session.get.return_value = None

    result = await repository.get(uuid4())

    assert result is None


@pytest.mark.asyncio
async def test_update(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Update should modify only supplied fields."""

    memory = Memory(
        id=uuid4(),
        content="Old content",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.5,
        metadata_={},
    )

    session.get.return_value = memory

    data = MemoryUpdate(
        content="New content",
        importance=0.9,
    )

    result = await repository.update(
        memory.id,
        data,
    )

    assert result is not None
    assert result.content == "New content"
    assert result.importance == 0.9
    assert memory.memory_type is MemoryType.PREFERENCE
    assert memory.status is MemoryStatus.ACTIVE

    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_returns_none(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Update should return None for a missing memory."""

    session.get.return_value = None

    result = await repository.update(
        uuid4(),
        MemoryUpdate(content="New content"),
    )

    assert result is None
    session.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_existing_memory(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Delete should remove an existing memory."""

    memory = Memory(
        id=uuid4(),
        content="Delete me.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.4,
        metadata_={},
    )

    session.get.return_value = memory

    result = await repository.delete(memory.id)

    assert result is True

    session.delete.assert_awaited_once_with(memory)


@pytest.mark.asyncio
async def test_delete_missing_memory(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """Delete should return False for a missing memory."""

    session.get.return_value = None

    result = await repository.delete(uuid4())

    assert result is False
    session.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_list(
    repository: MemoryRepository,
    session: MagicMock,
) -> None:
    """List should return serialized memories."""

    memory = Memory(
        id=uuid4(),
        content="A memory.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
        status=MemoryStatus.ACTIVE,
        importance=0.5,
        metadata_={},
    )

    scalars = MagicMock()
    scalars.all.return_value = [memory]

    result = MagicMock()
    result.scalars.return_value = scalars

    session.execute.return_value = result

    output = await repository.list(
        limit=25,
        offset=5,
    )

    assert len(output) == 1
    assert output[0].id == memory.id
    assert output[0].content == "A memory."

    session.execute.assert_awaited_once()

    statement = session.execute.await_args.args[0]

    assert statement._limit_clause.value == 25
    assert statement._offset_clause.value == 5