"""Redis client."""

from __future__ import annotations

from redis.asyncio import ConnectionPool
from redis.asyncio import Redis

from app.core.settings import settings


def create_connection_pool() -> ConnectionPool:
    """Create the Redis connection pool."""

    return ConnectionPool.from_url(
        settings.redis.url,
        decode_responses=settings.redis.decode_responses,
        socket_timeout=settings.redis.socket_timeout,
        max_connections=50,
    )


pool = create_connection_pool()


def create_redis_client() -> Redis:
    """Create the Redis client."""

    return Redis(
        connection_pool=pool,
    )


redis_client = create_redis_client()