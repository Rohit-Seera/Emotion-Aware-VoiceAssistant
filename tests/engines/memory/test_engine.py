"""Tests for the memory engine."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.engines.memory.engine import MemoryEngine
from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryRead,
    MemorySearchRequest,
    MemorySearchResult,
)
from app.engines.memory.types import (
    MemorySource,
    MemoryType,
)


@pytest.fixture
def repository() -> MagicMock:
    """Return a mocked memory repository."""

    repository = MagicMock()
    repository.create = AsyncMock()
    repository.get = AsyncMock()
    repository.delete = AsyncMock()

    return repository


@pytest.fixture
def retriever() -> MagicMock:
    """Return a mocked memory retriever."""

    retriever = MagicMock()
    retriever.retrieve = AsyncMock()

    return retriever


@pytest.fixture
def ranker() -> MagicMock:
    """Return a mocked memory ranker."""

    return MagicMock()


@pytest.fixture
def selector() -> MagicMock:
    """Return a mocked memory selector."""

    return MagicMock()


@pytest.fixture
def importance() -> MagicMock:
    """Return a mocked importance calculator."""

    return MagicMock()


@pytest.fixture
def summarizer() -> MagicMock:
    """Return a mocked memory summarizer."""

    summarizer = MagicMock()
    summarizer.summarize = AsyncMock()

    return summarizer


@pytest.fixture
def engine(
    repository: MagicMock,
    retriever: MagicMock,
    ranker: MagicMock,
    selector: MagicMock,
    importance: MagicMock,
    summarizer: MagicMock,
) -> MemoryEngine:
    """Return a configured memory engine."""

    return MemoryEngine(
        repository=repository,
        retriever=retriever,
        ranker=ranker,
        selector=selector,
        importance=importance,
        summarizer=summarizer,
    )


@pytest.mark.asyncio
async def test_create_calculates_importance_and_persists(
    engine: MemoryEngine,
    repository: MagicMock,
    importance: MagicMock,
) -> None:
    """Create should calculate importance before persistence."""

    memory = MemoryCreate(
        content="User likes Python.",
        memory_type=MemoryType.PREFERENCE,
        source=MemorySource.CONVERSATION,
    )

    importance.calculate.return_value = 0.9

    saved = MagicMock(spec=MemoryRead)
    repository.create.return_value = saved

    result = await engine.create(memory)

    importance.calculate.assert_called_once_with(
        "User likes Python.",
        memory_type=MemoryType.PREFERENCE,
    )

    repository.create.assert_awaited_once()

    persisted = repository.create.await_args.args[0]

    assert persisted.content == memory.content
    assert persisted.memory_type == memory.memory_type
    assert persisted.source == memory.source
    assert persisted.importance == 0.9

    assert result is saved


@pytest.mark.asyncio
async def test_get_delegates_to_repository(
    engine: MemoryEngine,
    repository: MagicMock,
) -> None:
    """Get should delegate to the repository."""

    memory_id = uuid4()

    expected = MagicMock(spec=MemoryRead)
    repository.get.return_value = expected

    result = await engine.get(memory_id)

    repository.get.assert_awaited_once_with(
        memory_id,
    )

    assert result is expected


@pytest.mark.asyncio
async def test_delete_delegates_to_repository(
    engine: MemoryEngine,
    repository: MagicMock,
) -> None:
    """Delete should delegate to the repository."""

    memory_id = uuid4()

    repository.delete.return_value = True

    result = await engine.delete(memory_id)

    repository.delete.assert_awaited_once_with(
        memory_id,
    )

    assert result is True


@pytest.mark.asyncio
async def test_search_runs_retrieval_ranking_and_selection(
    engine: MemoryEngine,
    retriever: MagicMock,
    ranker: MagicMock,
    selector: MagicMock,
) -> None:
    """Search should run the complete retrieval pipeline."""

    request = MemorySearchRequest(
        query="What does the user like?",
        top_k=5,
        similarity_threshold=0.8,
    )

    retrieved = [
        MagicMock(spec=MemorySearchResult),
        MagicMock(spec=MemorySearchResult),
    ]

    ranked = [
        retrieved[1],
        retrieved[0],
    ]

    selected = [
        ranked[0],
    ]

    retriever.retrieve.return_value = retrieved
    ranker.rank.return_value = ranked
    selector.select.return_value = selected

    result = await engine.search(request)

    retriever.retrieve.assert_awaited_once_with(
        "What does the user like?",
        top_k=5,
        similarity_threshold=0.8,
    )

    ranker.rank.assert_called_once_with(
        retrieved,
    )

    selector.select.assert_called_once_with(
        ranked,
        top_k=5,
    )

    assert result is selected


@pytest.mark.asyncio
async def test_search_returns_empty_when_nothing_is_retrieved(
    engine: MemoryEngine,
    retriever: MagicMock,
    ranker: MagicMock,
    selector: MagicMock,
) -> None:
    """Search should handle an empty retrieval result."""

    request = MemorySearchRequest(
        query="Nothing",
    )

    retriever.retrieve.return_value = []
    ranker.rank.return_value = []
    selector.select.return_value = []

    result = await engine.search(request)

    assert result == []

    retriever.retrieve.assert_awaited_once_with(
        "Nothing",
        top_k=10,
        similarity_threshold=0.75,
    )

    ranker.rank.assert_called_once_with([])

    selector.select.assert_called_once_with(
        [],
        top_k=10,
    )


@pytest.mark.asyncio
async def test_summarize_delegates_to_summarizer(
    engine: MemoryEngine,
    summarizer: MagicMock,
) -> None:
    """Summarize should delegate to the summarizer."""

    summarizer.summarize.return_value = (
        "User prefers Python."
    )

    result = await engine.summarize(
        "The user said they prefer Python."
    )

    summarizer.summarize.assert_awaited_once_with(
        "The user said they prefer Python."
    )

    assert result == "User prefers Python."