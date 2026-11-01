"""Cache health check without eager Redis dependency imports."""


async def check_cache_health() -> bool:
    """Return cache health; false when Redis support is unavailable."""

    try:
        from redis.exceptions import RedisError
        from app.infrastructure.cache.client import redis_client
    except ImportError:
        return False

    try:
        await redis_client.ping()
        return True
    except RedisError:
        return False
