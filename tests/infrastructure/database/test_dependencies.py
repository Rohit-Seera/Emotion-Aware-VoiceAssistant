"""Tests for database dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.database.dependencies import get_db


@pytest.mark.asyncio
async def test_get_db_yields_session_and_closes_it() -> None:
    """get_db should yield the session and close it afterward."""

    session = MagicMock()
    session.close = AsyncMock()

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.infrastructure.database.dependencies.SessionLocal",
        return_value=context,
    ):
        generator = get_db()
        yielded = await anext(generator)

        assert yielded is session

        await generator.aclose()

    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_db_closes_session_when_body_raises() -> None:
    """get_db should close the session when generator cleanup runs."""

    session = MagicMock()
    session.close = AsyncMock()

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "app.infrastructure.database.dependencies.SessionLocal",
        return_value=context,
    ):
        generator = get_db()
        yielded = await anext(generator)

        assert yielded is session

        await generator.aclose()

    session.close.assert_awaited_once()
