"""Interfaces for the memory engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryRead,
    MemorySearchResult,
    MemoryUpdate,
)
from app.engines.memory.types import MemoryType


class MemoryRepositoryProtocol(Protocol):
    """Persistence contract for user- and character-scoped memory storage."""

    async def create(
        self,
        *,
        user_id: UUID,
        character_id: UUID,
        data: MemoryCreate,
    ) -> MemoryRead:
        """Persist a new scoped memory."""
        ...

    async def get(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        character_id: UUID,
    ) -> MemoryRead | None:
        """Retrieve a memory within its scope."""
        ...

    async def update(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        character_id: UUID,
        data: MemoryUpdate,
    ) -> MemoryRead | None:
        """Update a memory within its scope."""
        ...

    async def delete(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        character_id: UUID,
    ) -> bool:
        """Delete a memory within its scope."""
        ...

    async def list(
        self,
        *,
        user_id: UUID,
        character_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRead]:
        """List memories within a user and character scope."""
        ...


class MemoryRetrieverProtocol(Protocol):
    """Contract for semantic memory retrieval."""

    async def retrieve(
        self,
        query: str,
        *,
        user_id: UUID,
        character_id: UUID,
        top_k: int = 10,
        similarity_threshold: float = 0.75,
    ) -> list[MemorySearchResult]:
        """Retrieve memories relevant to a scoped query."""
        ...


class MemoryRankerProtocol(Protocol):
    """Contract for ranking retrieved memories."""

    def rank(
        self,
        memories: Sequence[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Rank candidate memories."""
        ...


class MemorySelectorProtocol(Protocol):
    """Contract for selecting memories for context."""

    def select(
        self,
        memories: Sequence[MemorySearchResult],
        *,
        max_results: int = 20,
    ) -> list[MemorySearchResult]:
        """Select the final memories to include in context."""
        ...


class MemoryImportanceProtocol(Protocol):
    """Contract for calculating memory importance."""

    def calculate(
        self,
        content: str,
        *,
        memory_type: MemoryType,
    ) -> float:
        """Calculate the importance score of a memory."""
        ...


class MemorySummarizerProtocol(Protocol):
    """Contract for summarizing conversations or memories."""

    async def summarize(
        self,
        content: str,
    ) -> str:
        """Generate a concise summary."""
        ...


class MemoryConsolidatorProtocol(Protocol):
    """Contract for consolidating related memories."""

    async def consolidate(
        self,
        memories: Sequence[MemoryRead],
    ) -> Sequence[MemoryRead]:
        """Consolidate related memories."""
        ...