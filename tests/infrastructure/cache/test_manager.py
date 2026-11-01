import pytest
from app.infrastructure.cache.manager import CacheManager

@pytest.fixture
def manager(mock_redis_client):
    return CacheManager(mock_redis_client)

@pytest.mark.asyncio
async def test_get(manager, mock_redis_client):
    mock_redis_client.get.return_value = "hello"
    assert await manager.get("key") == "hello"
    mock_redis_client.get.assert_awaited_once_with("key")

@pytest.mark.asyncio
async def test_get_missing(manager, mock_redis_client):
    mock_redis_client.get.return_value = None
    assert await manager.get("missing") is None

@pytest.mark.asyncio
async def test_set_without_ttl(manager, mock_redis_client):
    mock_redis_client.set.return_value = True
    assert await manager.set("key", "value") is True
    mock_redis_client.set.assert_awaited_once_with(name="key", value="value", ex=None)

@pytest.mark.asyncio
async def test_set_with_ttl(manager, mock_redis_client):
    mock_redis_client.set.return_value = True
    assert await manager.set("key", "value", ttl=60) is True
    mock_redis_client.set.assert_awaited_once_with(name="key", value="value", ex=60)

@pytest.mark.asyncio
async def test_delete(manager, mock_redis_client):
    mock_redis_client.delete.return_value = 1
    assert await manager.delete("key") == 1
    mock_redis_client.delete.assert_awaited_once_with("key")

@pytest.mark.asyncio
async def test_exists_true(manager, mock_redis_client):
    mock_redis_client.exists.return_value = 1
    assert await manager.exists("key") is True

@pytest.mark.asyncio
async def test_exists_false(manager, mock_redis_client):
    mock_redis_client.exists.return_value = 0
    assert await manager.exists("key") is False

@pytest.mark.asyncio
async def test_expire(manager, mock_redis_client):
    mock_redis_client.expire.return_value = True
    assert await manager.expire("key", 120) is True
    mock_redis_client.expire.assert_awaited_once_with("key", 120)

@pytest.mark.asyncio
async def test_clear(manager, mock_redis_client):
    await manager.clear()
    mock_redis_client.flushdb.assert_awaited_once_with()
