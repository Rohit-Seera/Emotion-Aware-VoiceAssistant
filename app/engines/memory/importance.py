"""Importance scoring for the memory engine."""

from __future__ import annotations

from app.engines.memory.types import MemoryType


class MemoryImportance:
    """Calculate a normalized importance score for a memory."""

    _TYPE_WEIGHTS: dict[MemoryType, float] = {
        MemoryType.IDENTITY: 0.90,
        MemoryType.GOAL: 0.85,
        MemoryType.RELATIONSHIP_EVENT: 0.80,
        MemoryType.PREFERENCE: 0.70,
        MemoryType.EMOTIONAL_EVENT: 0.75,
        MemoryType.CONVERSATION_SUMMARY: 0.60,
    }

    def calculate(
        self,
        content: str,
        *,
        memory_type: MemoryType,
    ) -> float:
        """Calculate an importance score between 0 and 1."""

        normalized = content.strip()

        if not normalized:
            return 0.0

        base_score = self._TYPE_WEIGHTS.get(
            memory_type,
            0.5,
        )

        # Slightly increase importance for richer memories,
        # while keeping the result normalized to [0, 1].
        word_count = len(normalized.split())

        if word_count >= 20:
            base_score += 0.05
        elif word_count >= 10:
            base_score += 0.025

        return min(max(base_score, 0.0), 1.0)