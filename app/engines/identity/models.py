"""Database models for user identity."""

from __future__ import annotations

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import TimestampMixin, UUIDMixin


class User(Base, UUIDMixin, TimestampMixin):
    """Persistent application user mapped to an authentication subject."""

    __tablename__ = "users"

    auth_subject: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )