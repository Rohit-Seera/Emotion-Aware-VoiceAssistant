"""Type definitions for the memory engine."""

from __future__ import annotations

from enum import StrEnum


class MemoryType(StrEnum):
    """Supported categories of memories."""

    IDENTITY = "identity"
    PREFERENCE = "preference"
    GOAL = "goal"
    EMOTIONAL_EVENT = "emotional_event"
    RELATIONSHIP_EVENT = "relationship_event"
    CONVERSATION_SUMMARY = "conversation_summary"


class MemoryStatus(StrEnum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MemorySource(StrEnum):
    """Origin of a stored memory."""

    CONVERSATION = "conversation"
    USER = "user"
    SYSTEM = "system"