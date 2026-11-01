"""Tests for database session management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from app.infrastructure.database.session import (
    SessionLocal,
    create_session,
)


def test_session_local_is_configured() -> None:
    """SessionLocal should be an async session factory."""

    assert SessionLocal.class_.__name__ == "AsyncSession"
    assert SessionLocal.kw.get("autoflush") is False
    assert SessionLocal.kw.get("expire_on_commit") is False


async def test_create_session_returns_session() -> None:
    """create_session should return a new AsyncSession."""

    fake_session = MagicMock()

    with patch(
        "app.infrastructure.database.session.SessionLocal",
        return_value=fake_session,
    ) as session_local:
        result = await create_session()

    assert result is fake_session
    session_local.assert_called_once_with()
