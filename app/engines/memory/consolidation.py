"""Memory consolidation."""

from __future__ import annotations

from collections.abc import Sequence

from app.engines.memory.schemas import MemoryRead


class MemoryConsolidator:
    """Consolidate duplicate memories deterministically."""

    async def consolidate(
        self,
        memories: Sequence[MemoryRead],
    ) -> Sequence[MemoryRead]:
        """Remove duplicate memories and retain the strongest version."""

        if not memories:
            return []

        consolidated: dict[
            tuple[str, object],
            MemoryRead,
        ] = {}

        for memory in memories:
            key = (
                memory.memory_type,
                memory.content.strip().casefold(),
            )

            existing = consolidated.get(key)

            if existing is None:
                consolidated[key] = memory
                continue

            if memory.importance > existing.importance:
                consolidated[key] = memory

        return list(consolidated.values())