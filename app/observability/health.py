"""Application health checks."""

from __future__ import annotations

from app.infrastructure.cache.health import (
    check_cache_health,
)
from app.infrastructure.database.health import (
    check_database_health,
)


async def check_application_health() -> dict[str, bool]:
    """Return the health status of application dependencies."""

    database = await check_database_health()
    cache = await check_cache_health()

    return {
        "database": database,
        "cache": cache,
        "healthy": database and cache,
    }


async def is_application_healthy() -> bool:
    """Return whether all required dependencies are healthy."""

    health = await check_application_health()

    return health["healthy"]