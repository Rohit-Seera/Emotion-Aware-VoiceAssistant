"""Redis cache infrastructure with lazy public exports."""

__all__ = ["CacheManager", "cache", "get_cache", "redis_client"]


def __getattr__(name: str):
    if name == "redis_client":
        from .client import redis_client
        return redis_client
    if name == "get_cache":
        from .dependencies import get_cache
        return get_cache
    if name in {"CacheManager", "cache"}:
        from .manager import CacheManager, cache
        return {"CacheManager": CacheManager, "cache": cache}[name]
    raise AttributeError(name)
