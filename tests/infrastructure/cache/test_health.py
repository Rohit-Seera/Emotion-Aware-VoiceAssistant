from unittest.mock import AsyncMock, patch
import pytest
from redis.exceptions import RedisError
from app.infrastructure.cache.health import check_cache_health

@pytest.mark.asyncio
async def test_health_check_success():
    ping = AsyncMock(return_value=True)
    with patch("app.infrastructure.cache.health.redis_client") as client:
        client.ping = ping
        assert await check_cache_health() is True
    ping.assert_awaited_once_with()

@pytest.mark.asyncio
async def test_health_check_redis_error():
    ping = AsyncMock(side_effect=RedisError("unavailable"))
    with patch("app.infrastructure.cache.health.redis_client") as client:
        client.ping = ping
        assert await check_cache_health() is False
