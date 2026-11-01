from unittest.mock import AsyncMock, MagicMock
import pytest

@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    client.collection_exists = AsyncMock()
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    return client

@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed = AsyncMock()
    provider.embed_many = AsyncMock()
    return provider
