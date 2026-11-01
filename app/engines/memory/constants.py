"""Constants used by the memory engine."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Memory categories
# ---------------------------------------------------------------------------

MEMORY_IDENTITY = "identity"
MEMORY_PREFERENCE = "preference"
MEMORY_GOAL = "goal"
MEMORY_EMOTIONAL_EVENT = "emotional_event"
MEMORY_RELATIONSHIP_EVENT = "relationship_event"
MEMORY_CONVERSATION_SUMMARY = "conversation_summary"


MEMORY_TYPES = (
    MEMORY_IDENTITY,
    MEMORY_PREFERENCE,
    MEMORY_GOAL,
    MEMORY_EMOTIONAL_EVENT,
    MEMORY_RELATIONSHIP_EVENT,
    MEMORY_CONVERSATION_SUMMARY,
)


# ---------------------------------------------------------------------------
# Retrieval defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 10
DEFAULT_MAX_RESULTS = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Consolidation / summarization defaults
# ---------------------------------------------------------------------------

DEFAULT_SUMMARIZATION_THRESHOLD = 30
DEFAULT_CONSOLIDATION_INTERVAL = 3600


# ---------------------------------------------------------------------------
# Embedding defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_DIMENSION = 1536


# ---------------------------------------------------------------------------
# Memory status
# ---------------------------------------------------------------------------

MEMORY_STATUS_ACTIVE = "active"
MEMORY_STATUS_ARCHIVED = "archived"


MEMORY_STATUSES = (
    MEMORY_STATUS_ACTIVE,
    MEMORY_STATUS_ARCHIVED,
)


# ---------------------------------------------------------------------------
# Memory source
# ---------------------------------------------------------------------------

MEMORY_SOURCE_CONVERSATION = "conversation"
MEMORY_SOURCE_USER = "user"
MEMORY_SOURCE_SYSTEM = "system"


MEMORY_SOURCES = (
    MEMORY_SOURCE_CONVERSATION,
    MEMORY_SOURCE_USER,
    MEMORY_SOURCE_SYSTEM,
)