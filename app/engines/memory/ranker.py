"""Memory search result ranking."""

from __future__ import annotations

from collections.abc import Sequence

from app.engines.memory.schemas import MemorySearchResult


class MemoryRanker:
    """Rank memory search results by relevance score."""

    def rank(
        self,
        memories: Sequence[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Return memories ordered from highest to lowest score."""

        return sorted(
            memories,
            key=lambda memory: memory.score,
            reverse=True,
        )