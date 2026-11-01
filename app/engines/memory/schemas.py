"""Pydantic schemas for the memory engine."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)


class MemoryCreate(BaseModel):
    """Data required to create a user- and character-scoped memory."""

    user_id: UUID
    character_id: UUID

    content: str = Field(min_length=1)

    memory_type: MemoryType

    source: MemorySource = MemorySource.CONVERSATION

    importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class MemoryUpdate(BaseModel):
    """Data that can be changed on an existing memory."""

    content: str | None = Field(
        default=None,
        min_length=1,
    )

    memory_type: MemoryType | None = None

    status: MemoryStatus | None = None

    importance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    metadata: dict[str, object] | None = None


class MemoryRead(BaseModel):
    """Serialized representation of a stored memory."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )

    id: UUID

    user_id: UUID
    character_id: UUID

    content: str

    memory_type: MemoryType

    source: MemorySource

    status: MemoryStatus

    importance: float

    metadata: dict[str, object] = Field(
        validation_alias="metadata_",
    )

    created_at: datetime
    updated_at: datetime


class MemorySearchResult(BaseModel):
    """Memory returned from semantic retrieval."""

    memory: MemoryRead

    score: float = Field(
        ge=0.0,
        le=1.0,
    )


class MemorySearchRequest(BaseModel):
    """Parameters for searching memories."""

    user_id: UUID
    character_id: UUID

    query: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=10,
        ge=1,
    )

    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
    )