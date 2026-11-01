"""Application health endpoints."""

from fastapi import APIRouter, Response, status

from app.infrastructure.cache.health import check_cache_health
from app.infrastructure.database.health import check_database_health

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("")
async def health_check() -> dict[str, object]:
    """Return the health status of application dependencies."""

    database_healthy = await check_database_health()
    cache_healthy = await check_cache_health()

    healthy = database_healthy and cache_healthy

    return {
        "status": "healthy" if healthy else "unhealthy",
        "database": database_healthy,
        "cache": cache_healthy,
    }


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Return whether the process can receive traffic."""

    return {"status": "alive"}


@router.get("/ready")
async def readiness_check(response: Response) -> dict[str, object]:
    """Return dependency readiness with a useful deployment status code."""

    database_healthy = await check_database_health()
    cache_healthy = await check_cache_health()
    ready = database_healthy and cache_healthy

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if ready else "not_ready",
        "database": database_healthy,
        "cache": cache_healthy,
    }
