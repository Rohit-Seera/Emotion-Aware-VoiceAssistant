"""Database models for persistent companion characters."""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import TimestampMixin, UUIDMixin


class Character(Base, UUIDMixin, TimestampMixin):
    """Persistent companion character definition."""

    __tablename__ = "characters"

    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    personality: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    voice_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )