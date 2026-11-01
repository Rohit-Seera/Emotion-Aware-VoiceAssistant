"""Tests for the application API router."""

from app.api.router import router
from app.api.v1.router import router as v1_router


def test_api_router_has_api_prefix() -> None:
    """Application API router should use the /api prefix."""

    assert router.prefix == "/api"


def test_v1_router_has_v1_prefix() -> None:
    """Version 1 router should use the /v1 prefix."""

    assert v1_router.prefix == "/v1"


def test_v1_router_is_registered() -> None:
    """Version 1 router should be registered with the API router."""

    assert len(router.routes) == 1

    included_router = router.routes[0]

    assert included_router is not None