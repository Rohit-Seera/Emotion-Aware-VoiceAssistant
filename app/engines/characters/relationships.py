"""Database model for user-character relationships."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import TimestampMixin, UUIDMixin


class CharacterRelationship(Base, UUIDMixin, TimestampMixin):
    """User-specific relationship state for a companion character."""

    __tablename__ = "character_relationships"

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

    relationship_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )