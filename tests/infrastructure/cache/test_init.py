import app.infrastructure.cache as package
from app.infrastructure.cache.client import redis_client
from app.infrastructure.cache.dependencies import get_cache
from app.infrastructure.cache.manager import CacheManager, cache

def test_public_exports():
    assert package.CacheManager is CacheManager
    assert package.cache is cache
    assert package.get_cache is get_cache
    assert package.redis_client is redis_client

def test_all_exports():
    assert set(package.__all__) == {"CacheManager", "cache", "get_cache", "redis_client"}
