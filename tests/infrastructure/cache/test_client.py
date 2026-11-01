from importlib import import_module
from unittest.mock import MagicMock, patch

client_module = import_module("app.infrastructure.cache.client")

def test_create_connection_pool_uses_settings():
    fake_pool = MagicMock()
    with (
        patch("app.infrastructure.cache.client.ConnectionPool.from_url", return_value=fake_pool) as from_url,
        patch("app.infrastructure.cache.client.settings") as settings,
    ):
        settings.redis.url = "redis://localhost:6379/0"
        settings.redis.decode_responses = True
        settings.redis.socket_timeout = 5.0
        result = client_module.create_connection_pool()
    assert result is fake_pool
    from_url.assert_called_once_with(
        "redis://localhost:6379/0",
        decode_responses=True,
        socket_timeout=5.0,
        max_connections=50,
    )

def test_create_redis_client_uses_global_pool():
    fake_client = MagicMock()
    with patch("app.infrastructure.cache.client.Redis", return_value=fake_client) as redis:
        result = client_module.create_redis_client()
    assert result is fake_client
    redis.assert_called_once_with(connection_pool=client_module.pool)

def test_global_redis_client_exists():
    assert client_module.redis_client is not None
