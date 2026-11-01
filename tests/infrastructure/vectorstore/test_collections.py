from unittest.mock import AsyncMock, patch

import pytest
from qdrant_client.http.models import Distance, VectorParams

from app.infrastructure.vectorstore.collections import CollectionManager


@pytest.fixture
def manager():
    return CollectionManager()


@pytest.mark.asyncio
async def test_name_uses_configured_collection(manager):
    with patch(
        "app.infrastructure.vectorstore.collections.settings"
    ) as settings:
        settings.qdrant.collection = "weings-memory"
        assert manager.name == "weings-memory"


@pytest.mark.asyncio
async def test_exists_returns_qdrant_result(manager, mock_qdrant_client):
    mock_qdrant_client.collection_exists.return_value = True
    with patch(
        "app.infrastructure.vectorstore.collections.qdrant_client",
        mock_qdrant_client,
    ):
        result = await manager.exists()

    assert result is True
    mock_qdrant_client.collection_exists.assert_awaited_once_with(
        manager.name,
    )


@pytest.mark.asyncio
async def test_create_skips_when_collection_exists(
    manager,
    mock_qdrant_client,
):
    mock_qdrant_client.collection_exists.return_value = True
    with patch(
        "app.infrastructure.vectorstore.collections.qdrant_client",
        mock_qdrant_client,
    ):
        await manager.create()

    mock_qdrant_client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_creates_collection_with_config(
    manager,
    mock_qdrant_client,
):
    mock_qdrant_client.collection_exists.return_value = False

    with (
        patch(
            "app.infrastructure.vectorstore.collections.qdrant_client",
            mock_qdrant_client,
        ),
        patch(
            "app.infrastructure.vectorstore.collections.settings"
        ) as settings,
    ):
        settings.qdrant.collection = "weings-memory"
        settings.memory.embedding_dimension = 1536

        await manager.create()

    mock_qdrant_client.create_collection.assert_awaited_once()
    kwargs = mock_qdrant_client.create_collection.await_args.kwargs
    assert kwargs["collection_name"] == "weings-memory"

    vectors = kwargs["vectors_config"]
    assert isinstance(vectors, VectorParams)
    assert vectors.size == 1536
    assert vectors.distance == Distance.COSINE


@pytest.mark.asyncio
async def test_delete_skips_when_collection_missing(
    manager,
    mock_qdrant_client,
):
    mock_qdrant_client.collection_exists.return_value = False

    with patch(
        "app.infrastructure.vectorstore.collections.qdrant_client",
        mock_qdrant_client,
    ):
        await manager.delete()

    mock_qdrant_client.delete_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_deletes_existing_collection(
    manager,
    mock_qdrant_client,
):
    mock_qdrant_client.collection_exists.return_value = True

    with patch(
        "app.infrastructure.vectorstore.collections.qdrant_client",
        mock_qdrant_client,
    ):
        await manager.delete()

    mock_qdrant_client.delete_collection.assert_awaited_once_with(
        collection_name=manager.name,
    )
