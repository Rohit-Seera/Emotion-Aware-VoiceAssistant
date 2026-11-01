"""Tests for memory engine interfaces."""

from collections.abc import Sequence
from typing import get_type_hints

import pytest

from app.engines.memory.interfaces import (
    MemoryConsolidatorProtocol,
    MemoryImportanceProtocol,
    MemoryRankerProtocol,
    MemoryRepositoryProtocol,
    MemoryRetrieverProtocol,
    MemorySelectorProtocol,
    MemorySummarizerProtocol,
)


@pytest.mark.unit
def test_repository_protocol_is_protocol() -> None:
    """Repository interface should be a Protocol."""

    assert getattr(
        MemoryRepositoryProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_retriever_protocol_is_protocol() -> None:
    """Retriever interface should be a Protocol."""

    assert getattr(
        MemoryRetrieverProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_ranker_protocol_is_protocol() -> None:
    """Ranker interface should be a Protocol."""

    assert getattr(
        MemoryRankerProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_selector_protocol_is_protocol() -> None:
    """Selector interface should be a Protocol."""

    assert getattr(
        MemorySelectorProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_importance_protocol_is_protocol() -> None:
    """Importance interface should be a Protocol."""

    assert getattr(
        MemoryImportanceProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_summarizer_protocol_is_protocol() -> None:
    """Summarizer interface should be a Protocol."""

    assert getattr(
        MemorySummarizerProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_consolidator_protocol_is_protocol() -> None:
    """Consolidator interface should be a Protocol."""

    assert getattr(
        MemoryConsolidatorProtocol,
        "_is_protocol",
        False,
    )


@pytest.mark.unit
def test_repository_methods_exist() -> None:
    """Repository should expose the persistence contract."""

    assert hasattr(MemoryRepositoryProtocol, "create")
    assert hasattr(MemoryRepositoryProtocol, "get")
    assert hasattr(MemoryRepositoryProtocol, "update")
    assert hasattr(MemoryRepositoryProtocol, "delete")
    assert hasattr(MemoryRepositoryProtocol, "list")


@pytest.mark.unit
def test_retrieval_pipeline_methods_exist() -> None:
    """Retrieval pipeline interfaces should expose their methods."""

    assert hasattr(MemoryRetrieverProtocol, "retrieve")
    assert hasattr(MemoryRankerProtocol, "rank")
    assert hasattr(MemorySelectorProtocol, "select")


@pytest.mark.unit
def test_processing_methods_exist() -> None:
    """Processing interfaces should expose their methods."""

    assert hasattr(MemoryImportanceProtocol, "calculate")
    assert hasattr(MemorySummarizerProtocol, "summarize")
    assert hasattr(MemoryConsolidatorProtocol, "consolidate")


@pytest.mark.unit
def test_retriever_signature() -> None:
    """Retriever should accept query and retrieval controls."""

    hints = get_type_hints(
        MemoryRetrieverProtocol.retrieve,
    )

    assert "query" in hints
    assert "top_k" in hints
    assert "similarity_threshold" in hints


@pytest.mark.unit
@pytest.mark.unit
def test_ranker_accepts_sequence() -> None:
    """Ranker should operate on a sequence of search results."""

    hints = get_type_hints(
        MemoryRankerProtocol.rank,
    )

    assert "memories" in hints