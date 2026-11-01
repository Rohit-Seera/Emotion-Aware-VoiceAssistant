import pytest
from unittest.mock import patch

from app.infrastructure.vectorstore.embeddings import EmbeddingManager


@pytest.fixture
def manager(mock_embedding_provider):
    return EmbeddingManager(mock_embedding_provider)


def test_provider_returns_configured_provider(
    manager,
    mock_embedding_provider,
):
    assert manager.provider is mock_embedding_provider


def test_provider_raises_when_not_configured():
    manager = EmbeddingManager()
    with pytest.raises(
        RuntimeError,
        match="Embedding provider has not been configured",
    ):
        _ = manager.provider


def test_dimension_uses_memory_setting(manager):
    with patch(
        "app.infrastructure.vectorstore.embeddings.settings"
    ) as settings:
        settings.memory.embedding_dimension = 768
        assert manager.dimension == 768


@pytest.mark.asyncio
async def test_embed(manager, mock_embedding_provider):
    mock_embedding_provider.embed.return_value = [0.1, 0.2, 0.3]

    result = await manager.embed("Hello")

    assert result == [0.1, 0.2, 0.3]
    mock_embedding_provider.embed.assert_awaited_once_with("Hello")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
async def test_embed_rejects_empty_text(manager, text):
    with pytest.raises(ValueError, match="Text cannot be empty"):
        await manager.embed(text)


@pytest.mark.asyncio
async def test_embed_many(manager, mock_embedding_provider):
    expected = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedding_provider.embed_many.return_value = expected

    result = await manager.embed_many(["one", "two"])

    assert result == expected
    mock_embedding_provider.embed_many.assert_awaited_once_with(
        ["one", "two"],
    )


@pytest.mark.asyncio
async def test_embed_many_empty_input(manager, mock_embedding_provider):
    result = await manager.embed_many([])

    assert result == []
    mock_embedding_provider.embed_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["one", ""], ["   "], ["one", "\t"]])
async def test_embed_many_rejects_empty_values(manager, texts):
    with pytest.raises(
        ValueError,
        match="Texts cannot contain empty values",
    ):
        await manager.embed_many(texts)
