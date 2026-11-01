"""Shared types for AI providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProviderUsage:
    """Token usage reported by a provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class ProviderResponse:
    """Standard response returned by an AI provider."""

    content: str
    provider: str
    model: str

    usage: ProviderUsage | None = None

    finish_reason: str | None = None
    request_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderChunk:
    """A single chunk produced during streaming."""

    content: str = ""

    provider: str = ""
    model: str = ""

    request_id: str | None = None
    finish_reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingResult:
    """Standard embedding response."""

    embeddings: list[list[float]]

    provider: str
    model: str

    usage: ProviderUsage | None = None

    metadata: dict[str, Any] = field(default_factory=dict)