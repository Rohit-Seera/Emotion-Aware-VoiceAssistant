"""Cache dependencies."""

from collections.abc import AsyncGenerator

from app.infrastructure.cache.manager import CacheManager
from app.infrastructure.cache.manager import cache


async def get_cache() -> AsyncGenerator[CacheManager, None]:
    """Provide the application cache manager."""

    yield cache