"""Tests for semantic memory retrieval."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.engines.memory.retriever import MemoryRetriever
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


@pytest.fixture
def retriever() -> MemoryRetriever:
    """Return a memory retriever."""

    return MemoryRetriever()


def make_qdrant_result(
    *,
    score: float = 0.9,
) -> MagicMock:
    """Create a mocked Qdrant search result."""

    memory_id = uuid4()
    timestamp = datetime.now(timezone.utc)

    result = MagicMock()

    result.score = score
    result.payload = {
        "id": str(memory_id),
        "content": "User likes Python and AI.",
        "memory_type": MemoryType.PREFERENCE.value,
        "source": MemorySource.CONVERSATION.value,
        "status": MemoryStatus.ACTIVE.value,
        "importance": 0.8,
        "metadata": {
            "topic": "AI",
        },
        "created_at": timestamp.isoformat(),
        "updated_at": timestamp.isoformat(),
    }

    return result


def make_qdrant_response(
    result: MagicMock,
) -> MagicMock:
    """Create a mocked Qdrant query response."""

    response = MagicMock()
    response.points = [result]

    return response


@pytest.mark.asyncio
async def test_retrieve_embeds_query(
    retriever: MemoryRetriever,
) -> None:
    """Retriever should create an embedding for the query."""

    result = make_qdrant_result()
    response = make_qdrant_response(result)

    with (
        patch(
            "app.engines.memory.retriever.embeddings.embed",
            new=AsyncMock(
                return_value=[0.1, 0.2, 0.3],
            ),
        ) as embed,
        patch(
            "app.engines.memory.retriever.qdrant_client.query_points",
            new=AsyncMock(return_value=response),
        ),
    ):
        output = await retriever.retrieve(
            "What does the user like?"
        )

    embed.assert_awaited_once_with(
        "What does the user like?"
    )

    assert len(output) == 1


@pytest.mark.asyncio
async def test_retrieve_queries_correct_collection(
    retriever: MemoryRetriever,
) -> None:
    """Retriever should search the configured collection."""

    result = make_qdrant_result()
    response = make_qdrant_response(result)

    with (
        patch(
            "app.engines.memory.retriever.embeddings.embed",
            new=AsyncMock(
                return_value=[0.1, 0.2],
            ),
        ),
        patch(
            "app.engines.memory.retriever.qdrant_client.query_points",
            new=AsyncMock(return_value=response),
        ) as query_points,
    ):
        await retriever.retrieve(
            "test query",
            top_k=5,
            similarity_threshold=0.8,
        )

    query_points.assert_awaited_once_with(
        collection_name="weings-memory",
        query=[0.1, 0.2],
        limit=5,
        score_threshold=0.8,
    )


@pytest.mark.asyncio
async def test_retrieve_maps_qdrant_result(
    retriever: MemoryRetriever,
) -> None:
    """Retriever should map Qdrant results to memory search results."""

    result = make_qdrant_result(score=0.92)
    response = make_qdrant_response(result)

    with (
        patch(
            "app.engines.memory.retriever.embeddings.embed",
            new=AsyncMock(
                return_value=[0.1, 0.2],
            ),
        ),
        patch(
            "app.engines.memory.retriever.qdrant_client.query_points",
            new=AsyncMock(return_value=response),
        ),
    ):
        output = await retriever.retrieve(
            "AI preference"
        )

    assert len(output) == 1

    search_result = output[0]

    assert search_result.score == 0.92
    assert search_result.memory.content == (
        "User likes Python and AI."
    )
    assert search_result.memory.memory_type is (
        MemoryType.PREFERENCE
    )
    assert search_result.memory.source is (
        MemorySource.CONVERSATION
    )
    assert search_result.memory.status is (
        MemoryStatus.ACTIVE
    )
    assert search_result.memory.importance == 0.8
    assert search_result.memory.metadata == {
        "topic": "AI",
    }


@pytest.mark.asyncio
async def test_retrieve_empty_query_raises(
    retriever: MemoryRetriever,
) -> None:
    """Empty queries should be rejected."""

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        await retriever.retrieve("   ")


@pytest.mark.asyncio
async def test_retrieve_non_positive_top_k_returns_empty(
    retriever: MemoryRetriever,
) -> None:
    """Non-positive top_k should return no results."""

    with patch(
        "app.engines.memory.retriever.embeddings.embed",
        new=AsyncMock(),
    ) as embed:
        output = await retriever.retrieve(
            "test",
            top_k=0,
        )

    assert output == []
    embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_threshold(
    retriever: MemoryRetriever,
) -> None:
    """Similarity threshold must be between zero and one."""

    with pytest.raises(
        ValueError,
        match="Similarity threshold must be between 0 and 1",
    ):
        await retriever.retrieve(
            "test",
            similarity_threshold=1.5,
        )