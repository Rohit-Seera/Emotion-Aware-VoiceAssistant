"""Tests for application health checks."""

from unittest.mock import AsyncMock, patch

import pytest

from app.observability.health import (
    check_application_health,
    is_application_healthy,
)


@pytest.mark.asyncio
async def test_application_health_when_all_dependencies_are_healthy() -> None:
    """Application should be healthy when all dependencies are healthy."""

    with (
        patch(
            "app.observability.health.check_database_health",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "app.observability.health.check_cache_health",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await check_application_health()

    assert result == {
        "database": True,
        "cache": True,
        "healthy": True,
    }


@pytest.mark.asyncio
async def test_application_health_when_database_is_unhealthy() -> None:
    """Application should be unhealthy when database fails."""

    with (
        patch(
            "app.observability.health.check_database_health",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "app.observability.health.check_cache_health",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await check_application_health()

    assert result == {
        "database": False,
        "cache": True,
        "healthy": False,
    }


@pytest.mark.asyncio
async def test_application_health_when_cache_is_unhealthy() -> None:
    """Application should be unhealthy when cache fails."""

    with (
        patch(
            "app.observability.health.check_database_health",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "app.observability.health.check_cache_health",
            new=AsyncMock(return_value=False),
        ),
    ):
        result = await check_application_health()

    assert result == {
        "database": True,
        "cache": False,
        "healthy": False,
    }


@pytest.mark.asyncio
async def test_application_health_when_all_dependencies_fail() -> None:
    """Application should be unhealthy when all dependencies fail."""

    with (
        patch(
            "app.observability.health.check_database_health",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "app.observability.health.check_cache_health",
            new=AsyncMock(return_value=False),
        ),
    ):
        result = await check_application_health()

    assert result == {
        "database": False,
        "cache": False,
        "healthy": False,
    }


@pytest.mark.asyncio
async def test_is_application_healthy_returns_true() -> None:
    """Health helper should return true for a healthy application."""

    with patch(
        "app.observability.health.check_application_health",
        new=AsyncMock(
            return_value={
                "database": True,
                "cache": True,
                "healthy": True,
            },
        ),
    ):
        result = await is_application_healthy()

    assert result is True


@pytest.mark.asyncio
async def test_is_application_healthy_returns_false() -> None:
    """Health helper should return false for an unhealthy application."""

    with patch(
        "app.observability.health.check_application_health",
        new=AsyncMock(
            return_value={
                "database": False,
                "cache": True,
                "healthy": False,
            },
        ),
    ):
        result = await is_application_healthy()

    assert result is False