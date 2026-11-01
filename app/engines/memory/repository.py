"""Repository implementation for memory persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.engines.memory.models import Memory
from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryRead,
    MemoryUpdate,
)
from app.engines.memory.types import MemoryStatus
from app.infrastructure.repositories.base import BaseRepository


class MemoryRepository(BaseRepository[Memory]):
    """Repository for persisted user- and character-scoped memories."""

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        super().__init__(
            session=session,
            model=Memory,
        )

    @staticmethod
    def _to_read(
        memory: Memory,
    ) -> MemoryRead:
        """Convert a database model into a read schema."""

        now = datetime.now(timezone.utc)

        metadata = memory.metadata_

        if not isinstance(metadata, dict):
            metadata = {}

        return MemoryRead.model_validate(
            {
                "id": memory.id or uuid4(),
                "user_id": memory.user_id,
                "character_id": memory.character_id,
                "content": memory.content,
                "memory_type": memory.memory_type,
                "source": memory.source,
                "status": (
                    memory.status
                    or MemoryStatus.ACTIVE
                ),
                "importance": memory.importance,
                "metadata_": metadata,
                "created_at": (
                    memory.created_at
                    or now
                ),
                "updated_at": (
                    memory.updated_at
                    or now
                ),
            }
        )

    async def create(
        self,
        *,
        user_id: UUID,
        character_id: UUID,
        data: MemoryCreate,
    ) -> MemoryRead:
        """Create a memory scoped to a user and character."""

        now = datetime.now(timezone.utc)

        memory = Memory(
            id=uuid4(),
            user_id=user_id,
            character_id=character_id,
            content=data.content,
            memory_type=data.memory_type,
            source=data.source,
            status=MemoryStatus.ACTIVE,
            importance=data.importance,
            metadata_=data.metadata,
            created_at=now,
            updated_at=now,
        )

        await self.add(memory)

        return self._to_read(memory)

    async def get(
        self,
        *,
        memory_id: UUID | object,
        user_id: UUID,
        character_id: UUID,
    ) -> MemoryRead | None:
        """Get a memory only when it belongs to the user and character."""

        statement = (
            select(Memory)
            .where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
                Memory.character_id == character_id,
            )
        )

        result = await self.session.execute(statement)
        memory = result.scalar_one_or_none()

        if memory is None:
            return None

        return self._to_read(memory)

    async def update(
        self,
        *,
        memory_id: UUID | object,
        user_id: UUID,
        character_id: UUID,
        data: MemoryUpdate,
    ) -> MemoryRead | None:
        """Update a memory only when it belongs to the user and character."""

        statement = (
            select(Memory)
            .where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
                Memory.character_id == character_id,
            )
        )

        result = await self.session.execute(statement)
        memory = result.scalar_one_or_none()

        if memory is None:
            return None

        updates = data.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

        if "content" in updates:
            memory.content = updates["content"]

        if "memory_type" in updates:
            memory.memory_type = updates["memory_type"]

        if "status" in updates:
            memory.status = updates["status"]

        if "importance" in updates:
            memory.importance = updates["importance"]

        if "metadata" in updates:
            memory.metadata_ = updates["metadata"]

        memory.updated_at = datetime.now(timezone.utc)

        await self.session.flush()

        return self._to_read(memory)

    async def delete(
        self,
        *,
        memory_id: UUID | object,
        user_id: UUID,
        character_id: UUID,
    ) -> bool:
        """Delete a memory only when it belongs to the user and character."""

        statement = (
            select(Memory)
            .where(
                Memory.id == memory_id,
                Memory.user_id == user_id,
                Memory.character_id == character_id,
            )
        )

        result = await self.session.execute(statement)
        memory = result.scalar_one_or_none()

        if memory is None:
            return False

        await super().delete(memory)

        return True

    async def list(
        self,
        *,
        user_id: UUID,
        character_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRead]:
        """Return memories belonging only to the user and character."""

        statement = (
            select(Memory)
            .where(
                Memory.user_id == user_id,
                Memory.character_id == character_id,
            )
            .order_by(Memory.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        memories = list(result.scalars().all())

        return [
            self._to_read(memory)
            for memory in memories
        ]