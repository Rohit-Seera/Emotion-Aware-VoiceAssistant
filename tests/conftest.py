"""Shared fixtures for database tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_session() -> MagicMock:
    """Return a mocked SQLAlchemy async session."""

    session = MagicMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_engine() -> MagicMock:
    """Return a mocked SQLAlchemy async engine."""

    engine = MagicMock()
    engine.dispose = AsyncMock()
    return engine
