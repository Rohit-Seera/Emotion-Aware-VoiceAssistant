import pytest
from app.infrastructure.cache.dependencies import get_cache
from app.infrastructure.cache.manager import cache

@pytest.mark.asyncio
async def test_get_cache_yields_shared_cache():
    generator = get_cache()
    result = await anext(generator)
    assert result is cache
    await generator.aclose()
