"""Tests for the database declarative base."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import UUIDMixin


def test_base_is_declarative_base() -> None:
    """Base should be a SQLAlchemy declarative base."""

    assert Base.registry is not None
    assert Base.metadata is not None


def test_model_can_inherit_from_base() -> None:
    """ORM models should be able to inherit from Base."""

    table_name = f"test_models_{uuid4().hex}"

    class TestModel(UUIDMixin, Base):
        __tablename__ = table_name

        name: Mapped[str] = mapped_column(
            String(100),
            nullable=False,
        )

    assert TestModel.__tablename__ == table_name
    assert TestModel.__table__ in Base.metadata.tables.values()
    assert "id" in TestModel.__table__.columns
    assert "name" in TestModel.__table__.columns