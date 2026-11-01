"""Tests for the base repository."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.repositories.base import BaseRepository


class RepositoryTestModel(Base):
    """Minimal mapped model used for repository tests."""

    __tablename__ = "test_models"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
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
) -> BaseRepository[RepositoryTestModel]:
    """Return a repository using the test model."""

    return BaseRepository(
        session=session,
        model=RepositoryTestModel,
    )


@pytest.mark.asyncio
async def test_repository_stores_session_and_model(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """Repository should store its session and model."""

    assert repository.session is session
    assert repository.model is RepositoryTestModel


@pytest.mark.asyncio
async def test_get_by_id(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """get_by_id should use AsyncSession.get."""

    entity = RepositoryTestModel()
    session.get.return_value = entity

    result = await repository.get_by_id(1)

    assert result is entity

    session.get.assert_awaited_once_with(
        RepositoryTestModel,
        1,
    )


@pytest.mark.asyncio
async def test_get_by_id_returns_none(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """get_by_id should return None when entity is missing."""

    session.get.return_value = None

    result = await repository.get_by_id(999)

    assert result is None

    session.get.assert_awaited_once_with(
        RepositoryTestModel,
        999,
    )


@pytest.mark.asyncio
async def test_add(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """add should add the entity and flush the session."""

    entity = RepositoryTestModel()

    result = await repository.add(entity)

    assert result is entity

    session.add.assert_called_once_with(entity)
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """delete should delete the entity from the session."""

    entity = RepositoryTestModel()

    result = await repository.delete(entity)

    assert result is None

    session.delete.assert_awaited_once_with(entity)


@pytest.mark.asyncio
async def test_exists_true(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """exists should return True when entity exists."""

    entity = RepositoryTestModel()
    session.get.return_value = entity

    result = await repository.exists(1)

    assert result is True

    session.get.assert_awaited_once_with(
        RepositoryTestModel,
        1,
    )


@pytest.mark.asyncio
async def test_exists_false(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """exists should return False when entity does not exist."""

    session.get.return_value = None

    result = await repository.exists(999)

    assert result is False

    session.get.assert_awaited_once_with(
        RepositoryTestModel,
        999,
    )


@pytest.mark.asyncio
async def test_list_uses_default_pagination(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """list should use the default limit and offset."""

    entities = [
        RepositoryTestModel(),
        RepositoryTestModel(),
    ]

    scalars = MagicMock()
    scalars.all.return_value = entities

    result = MagicMock()
    result.scalars.return_value = scalars

    session.execute.return_value = result

    output = await repository.list()

    assert output == entities

    session.execute.assert_awaited_once()

    statement = session.execute.await_args.args[0]

    assert statement._limit_clause.value == 100
    assert statement._offset_clause.value == 0


@pytest.mark.asyncio
async def test_list_uses_custom_pagination(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """list should respect custom limit and offset."""

    entities = [RepositoryTestModel()]

    scalars = MagicMock()
    scalars.all.return_value = entities

    result = MagicMock()
    result.scalars.return_value = scalars

    session.execute.return_value = result

    output = await repository.list(
        limit=25,
        offset=10,
    )

    assert output == entities

    session.execute.assert_awaited_once()

    statement = session.execute.await_args.args[0]

    assert statement._limit_clause.value == 25
    assert statement._offset_clause.value == 10


@pytest.mark.asyncio
async def test_list_returns_empty_list(
    repository: BaseRepository[RepositoryTestModel],
    session: MagicMock,
) -> None:
    """list should return an empty list when no entities exist."""

    scalars = MagicMock()
    scalars.all.return_value = []

    result = MagicMock()
    result.scalars.return_value = scalars

    session.execute.return_value = result

    output = await repository.list()

    assert output == []