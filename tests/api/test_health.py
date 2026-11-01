"""Tests for health API endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.health import router


def create_test_app() -> FastAPI:
    """Create a minimal FastAPI application for testing."""

    app = FastAPI()
    app.include_router(router)

    return app


def test_health_returns_healthy_when_dependencies_are_healthy() -> None:
    """Health endpoint should report healthy when all dependencies work."""

    app = create_test_app()

    with (
        patch(
            "app.api.v1.health.check_database_health",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "app.api.v1.health.check_cache_health",
            new=AsyncMock(return_value=True),
        ),
    ):
        client = TestClient(app)

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "database": True,
        "cache": True,
    }


def test_health_returns_unhealthy_when_database_is_unhealthy() -> None:
    """Health endpoint should report unhealthy when database fails."""

    app = create_test_app()

    with (
        patch(
            "app.api.v1.health.check_database_health",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "app.api.v1.health.check_cache_health",
            new=AsyncMock(return_value=True),
        ),
    ):
        client = TestClient(app)

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unhealthy",
        "database": False,
        "cache": True,
    }


def test_health_returns_unhealthy_when_cache_is_unhealthy() -> None:
    """Health endpoint should report unhealthy when cache fails."""

    app = create_test_app()

    with (
        patch(
            "app.api.v1.health.check_database_health",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "app.api.v1.health.check_cache_health",
            new=AsyncMock(return_value=False),
        ),
    ):
        client = TestClient(app)

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "unhealthy",
        "database": True,
        "cache": False,
    }