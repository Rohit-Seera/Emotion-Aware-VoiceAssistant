"""Tests for memory engine constants."""

import pytest

from app.engines.memory.constants import (
    DEFAULT_CONSOLIDATION_INTERVAL,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MAX_RESULTS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SUMMARIZATION_THRESHOLD,
    DEFAULT_TOP_K,
    MEMORY_CONVERSATION_SUMMARY,
    MEMORY_EMOTIONAL_EVENT,
    MEMORY_GOAL,
    MEMORY_IDENTITY,
    MEMORY_PREFERENCE,
    MEMORY_RELATIONSHIP_EVENT,
    MEMORY_SOURCE_CONVERSATION,
    MEMORY_SOURCE_SYSTEM,
    MEMORY_SOURCE_USER,
    MEMORY_STATUS_ACTIVE,
    MEMORY_STATUS_ARCHIVED,
    MEMORY_STATUSES,
    MEMORY_SOURCES,
    MEMORY_TYPES,
)


@pytest.mark.unit
def test_memory_types() -> None:
    """Memory types should contain all supported categories."""

    assert MEMORY_TYPES == (
        MEMORY_IDENTITY,
        MEMORY_PREFERENCE,
        MEMORY_GOAL,
        MEMORY_EMOTIONAL_EVENT,
        MEMORY_RELATIONSHIP_EVENT,
        MEMORY_CONVERSATION_SUMMARY,
    )


@pytest.mark.unit
def test_retrieval_defaults() -> None:
    """Retrieval defaults should match memory settings."""

    assert DEFAULT_TOP_K == 10
    assert DEFAULT_MAX_RESULTS == 20
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.75


@pytest.mark.unit
def test_consolidation_defaults() -> None:
    """Consolidation defaults should match memory settings."""

    assert DEFAULT_SUMMARIZATION_THRESHOLD == 30
    assert DEFAULT_CONSOLIDATION_INTERVAL == 3600


@pytest.mark.unit
def test_embedding_default() -> None:
    """Embedding dimension should match the locked memory configuration."""

    assert DEFAULT_EMBEDDING_DIMENSION == 1536


@pytest.mark.unit
def test_memory_statuses() -> None:
    """Memory statuses should contain active and archived."""

    assert MEMORY_STATUSES == (
        MEMORY_STATUS_ACTIVE,
        MEMORY_STATUS_ARCHIVED,
    )


@pytest.mark.unit
def test_memory_sources() -> None:
    """Memory sources should contain supported origins."""

    assert MEMORY_SOURCES == (
        MEMORY_SOURCE_CONVERSATION,
        MEMORY_SOURCE_USER,
        MEMORY_SOURCE_SYSTEM,
    )