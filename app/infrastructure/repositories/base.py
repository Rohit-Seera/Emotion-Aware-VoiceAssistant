"""Base repository abstractions."""

from __future__ import annotations

from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.base import Base


ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Base repository for common database operations."""

    def __init__(
        self,
        session: AsyncSession,
        model: type[ModelT],
    ) -> None:
        self.session = session
        self.model = model

    async def get_by_id(
        self,
        entity_id: object,
    ) -> ModelT | None:
        """Return an entity by primary key."""

        return await self.session.get(
            self.model,
            entity_id,
        )

    async def add(
        self,
        entity: ModelT,
    ) -> ModelT:
        """Add an entity to the current session."""

        self.session.add(entity)
        await self.session.flush()

        return entity

    async def delete(
        self,
        entity: ModelT,
    ) -> None:
        """Delete an entity from the current session."""

        await self.session.delete(entity)

    async def exists(
        self,
        entity_id: object,
    ) -> bool:
        """Check whether an entity exists."""

        entity = await self.get_by_id(entity_id)

        return entity is not None

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelT]:
        """Return a paginated list of entities."""

        statement = (
            select(self.model)
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return list(result.scalars().all())