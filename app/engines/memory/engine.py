"""Memory engine orchestration."""

from __future__ import annotations

from uuid import UUID

from app.engines.memory.interfaces import (
    MemoryImportanceProtocol,
    MemoryRankerProtocol,
    MemoryRepositoryProtocol,
    MemoryRetrieverProtocol,
    MemorySelectorProtocol,
    MemorySummarizerProtocol,
)
from app.engines.memory.schemas import (
    MemoryCreate,
    MemoryRead,
    MemorySearchRequest,
    MemorySearchResult,
)


class MemoryEngine:
    """Coordinate user- and character-scoped memory operations."""

    def __init__(
        self,
        repository: MemoryRepositoryProtocol,
        retriever: MemoryRetrieverProtocol,
        ranker: MemoryRankerProtocol,
        selector: MemorySelectorProtocol,
        importance: MemoryImportanceProtocol,
        summarizer: MemorySummarizerProtocol,
    ) -> None:
        self.repository = repository
        self.retriever = retriever
        self.ranker = ranker
        self.selector = selector
        self.importance = importance
        self.summarizer = summarizer

    async def create(
        self,
        memory: MemoryCreate,
    ) -> MemoryRead:
        """Calculate importance and persist a new memory."""

        calculated_importance = self.importance.calculate(
            memory.content,
            memory_type=memory.memory_type,
        )

        memory = memory.model_copy(
            update={
                "importance": calculated_importance,
            },
        )

        if memory.character_id is None:
            raise ValueError(
                "character_id is required to create a memory."
            )

        return await self.repository.create(
            user_id=memory.user_id,
            character_id=memory.character_id,
            data=memory,
        )

    async def get(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        character_id: UUID,
    ) -> MemoryRead | None:
        """Retrieve a memory scoped to a user and character."""

        return await self.repository.get(
            memory_id=memory_id,
            user_id=user_id,
            character_id=character_id,
        )

    async def delete(
        self,
        *,
        memory_id: UUID,
        user_id: UUID,
        character_id: UUID,
    ) -> bool:
        """Delete a memory scoped to a user and character."""

        return await self.repository.delete(
            memory_id=memory_id,
            user_id=user_id,
            character_id=character_id,
        )

    async def search(
        self,
        request: MemorySearchRequest,
    ) -> list[MemorySearchResult]:
        """Retrieve, rank, and select relevant memories."""

        memories = await self.retriever.retrieve(
            request.query,
            user_id=request.user_id,
            character_id=request.character_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )

        ranked = self.ranker.rank(memories)

        return self.selector.select(
            ranked,
            max_results=request.top_k,
        )

    async def summarize(
        self,
        content: str,
    ) -> str:
        """Summarize supplied content."""

        return await self.summarizer.summarize(content)