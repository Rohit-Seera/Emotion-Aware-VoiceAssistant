"""Tests for the FastAPI application."""

from fastapi.testclient import TestClient

from app.main import app


def test_application_is_created() -> None:
    """FastAPI application should be created successfully."""

    assert app.title == "Emotion Aware Voice Assistant"
    assert app.version == "1.0.0"


def test_api_router_is_registered() -> None:
    """Application should expose the API router."""

    routes = {
        route.path
        for route in app.routes
        if hasattr(route, "path")
    }

    assert "/docs" in routes
    assert "/openapi.json" in routes

    openapi_routes = app.openapi()["paths"]

    assert "/api/v1/health" in openapi_routes

def test_openapi_is_available() -> None:
    """OpenAPI schema should be available."""

    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200

    data = response.json()

    assert data["info"]["title"] == "Emotion Aware Voice Assistant"
    assert "/api/v1/health" in data["paths"]