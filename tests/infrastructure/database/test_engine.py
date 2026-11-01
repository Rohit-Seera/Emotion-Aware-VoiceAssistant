"""Tests for database engine creation."""

from __future__ import annotations

from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest

engine_module = import_module(
    "app.infrastructure.database.engine"
)


def test_create_engine_uses_database_settings() -> None:
    """Engine creation should use the configured database settings."""

    fake_engine = MagicMock()

    with (
        patch(
            "app.infrastructure.database.engine.create_async_engine",
            return_value=fake_engine,
        ) as create_engine,
        patch(
            "app.infrastructure.database.engine.settings"
        ) as mock_settings,
    ):
        mock_settings.database.url = (
            "postgresql+asyncpg://test:test@localhost/test"
        )
        mock_settings.database.echo = False
        mock_settings.database.pool_size = 5
        mock_settings.database.max_overflow = 10
        mock_settings.database.pool_pre_ping = True
        mock_settings.database.pool_recycle = 1800

        result = engine_module.create_engine()

    assert result is fake_engine

    create_engine.assert_called_once_with(
    "postgresql+asyncpg://test:test@localhost/test",
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)