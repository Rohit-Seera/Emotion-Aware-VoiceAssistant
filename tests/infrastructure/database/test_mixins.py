"""Tests for database ORM mixins."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped

from app.infrastructure.database.base import Base
from app.infrastructure.database.mixins import (
    TimestampMixin,
    UUIDMixin,
)


def test_uuid_mixin_defines_primary_key() -> None:
    """UUIDMixin should provide a PostgreSQL UUID primary key."""

    class TestUUIDModel(UUIDMixin, Base):
        __tablename__ = "test_uuid_models"

    column = inspect(TestUUIDModel).columns.id

    assert column.primary_key is True
    assert isinstance(column.type, PostgreSQLUUID)
    assert column.type.as_uuid is True
    assert column.default is not None


def test_timestamp_mixin_defines_required_timestamps() -> None:
    """TimestampMixin should provide created and updated timestamps."""

    class TestTimestampModel(UUIDMixin, TimestampMixin, Base):
        __tablename__ = "test_timestamp_models"

    mapper = inspect(TestTimestampModel)
    created = mapper.columns.created_at
    updated = mapper.columns.updated_at

    assert created.nullable is False
    assert updated.nullable is False
    assert created.server_default is not None
    assert updated.server_default is not None
    assert updated.onupdate is not None


def test_uuid_default_is_callable() -> None:
    """UUIDMixin should configure a callable UUID default."""

    class TestUUIDDefaultModel(UUIDMixin, Base):
        __tablename__ = "test_uuid_default_models"

    default = inspect(TestUUIDDefaultModel).columns.id.default

    assert default is not None
    assert callable(default.arg)


def test_uuid_default_generates_uuid_via_function() -> None:
    """The UUID default function should generate UUID values."""

    class TestUUIDValueModel(UUIDMixin, Base):
        __tablename__ = "test_uuid_value_models"

    default = inspect(TestUUIDValueModel).columns.id.default

    value = default.arg(None)

    assert isinstance(value, UUID)


def test_timestamp_annotations_are_deferred_mapped_annotations() -> None:
    """TimestampMixin should expose the expected deferred annotations."""

    annotations = TimestampMixin.__annotations__

    assert annotations["created_at"] == "Mapped[datetime]"
    assert annotations["updated_at"] == "Mapped[datetime]"
