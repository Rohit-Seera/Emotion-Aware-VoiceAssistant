"""Base interface for AI providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
)


class BaseProvider(ABC):
    """Common interface implemented by all AI providers."""

    name: str

    @abstractmethod
    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a complete response."""

    @abstractmethod
    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderChunk]:
        """Stream a response incrementally."""

    @abstractmethod
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings for one or more texts."""

    async def health_check(self) -> bool:
        """Return whether the provider is currently available."""

        return True

    async def close(self) -> None:
        """Release provider resources."""

        return None