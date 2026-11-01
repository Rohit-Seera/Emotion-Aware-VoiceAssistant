"""Tests for database health checks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.infrastructure.database.health import check_database_health


@pytest.mark.asyncio
async def test_health_check_returns_true_when_query_succeeds() -> None:
    """Health check should return True when SELECT 1 succeeds."""

    session = MagicMock()
    session.execute = AsyncMock()

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.infrastructure.database.health.SessionLocal",
        return_value=context,
    ):
        result = await check_database_health()

    assert result is True
    session.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_check_returns_false_on_sqlalchemy_error() -> None:
    """Health check should return False for SQLAlchemy errors."""

    session = MagicMock()
    session.execute = AsyncMock(
        side_effect=SQLAlchemyError("database unavailable"),
    )

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.infrastructure.database.health.SessionLocal",
        return_value=context,
    ):
        result = await check_database_health()

    assert result is False
