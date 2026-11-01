"""Memory result selection."""

from __future__ import annotations

from collections.abc import Sequence

from app.engines.memory.schemas import MemorySearchResult


class MemorySelector:
    """Select the most relevant memories for the final context."""

    def select(
        self,
        memories: Sequence[MemorySearchResult],
        *,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        """Return at most ``top_k`` memories."""

        if top_k <= 0:
            return []

        return list(memories[:top_k])