"""Database models for the memory engine."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import TimestampMixin, UUIDMixin


class Memory(Base, UUIDMixin, TimestampMixin):
    """Persisted long-term memory scoped to a user and character."""

    __tablename__ = "memories"

    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    character_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("characters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    memory_type: Mapped[MemoryType] = mapped_column(
        String(64),
        nullable=False,
    )

    source: Mapped[MemorySource] = mapped_column(
        String(64),
        nullable=False,
        default=MemorySource.CONVERSATION,
    )

    status: Mapped[MemoryStatus] = mapped_column(
        String(32),
        nullable=False,
        default=MemoryStatus.ACTIVE,
    )

    importance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.5,
    )

    metadata_: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )