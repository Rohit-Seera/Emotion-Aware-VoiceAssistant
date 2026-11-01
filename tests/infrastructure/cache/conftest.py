from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.fixture
def mock_redis_client():
    client = MagicMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock()
    client.expire = AsyncMock()
    client.flushdb = AsyncMock()
    client.ping = AsyncMock()
    return client
