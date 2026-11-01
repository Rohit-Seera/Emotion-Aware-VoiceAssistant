"""Redis cache infrastructure helpers."""

from __future__ import annotations

from typing import Any

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool

from app.core.settings import settings


def create_connection_pool() -> ConnectionPool:
    """Create the shared Redis connection pool."""

    return ConnectionPool.from_url(
        settings.redis.url,
        decode_responses=settings.redis.decode_responses,
        socket_timeout=settings.redis.socket_timeout,
        max_connections=50,
    )


def create_redis_client(
    pool: ConnectionPool | None = None,
) -> Redis:
    """Create a Redis client using the shared connection pool."""

    connection_pool = (
        pool
        if pool is not None
        else globals()["pool"]
    )

    return Redis(
        connection_pool=connection_pool,
    )


pool = create_connection_pool()

redis_client: Redis = create_redis_client(pool)