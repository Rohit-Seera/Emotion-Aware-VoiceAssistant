"""Redis cache manager."""

from __future__ import annotations

from typing import Any

from redis.asyncio import Redis

from app.infrastructure.cache.client import redis_client


class CacheManager:
    """High-level interface for Redis operations."""

    def __init__(self, client: Redis) -> None:
        self._client = client

    async def get(self, key: str) -> str | None:
        """Retrieve a value from the cache."""

        return await self._client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        ttl: int | None = None,
    ) -> bool:
        """Store a value in the cache."""

        return await self._client.set(
            name=key,
            value=value,
            ex=ttl,
        )

    async def delete(self, key: str) -> int:
        """Remove a value from the cache."""

        return await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        """Check whether a key exists."""

        return bool(await self._client.exists(key))

    async def expire(
        self,
        key: str,
        ttl: int,
    ) -> bool:
        """Update the expiration time of a key."""

        return await self._client.expire(key, ttl)

    async def clear(self) -> None:
        """Remove all keys from the current Redis database."""

        await self._client.flushdb()


cache = CacheManager(redis_client)