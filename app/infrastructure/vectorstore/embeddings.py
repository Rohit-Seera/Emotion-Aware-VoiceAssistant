"""Embedding infrastructure."""

from __future__ import annotations

from collections.abc import Sequence

from app.core.settings import settings
from app.providers.manager.provider_manager import ProviderManager


class EmbeddingManager:
    """Manage text embeddings used by the vector store."""

    def __init__(
        self,
        provider_manager: ProviderManager | None = None,
    ) -> None:
        self._provider_manager = (
            provider_manager or ProviderManager()
        )

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""

        return settings.memory.embedding_dimension

    async def embed(
        self,
        text: str,
    ) -> list[float]:
        """Create an embedding for a single text."""

        if not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        result = await self._provider_manager.embed(
            [text],
            provider=settings.ai.embedding_provider,
        )

        if not result.embeddings:
            raise RuntimeError(
                "Embedding provider returned no embedding."
            )

        return result.embeddings[0]

    async def embed_many(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Create embeddings for multiple texts."""

        if not texts:
            return []

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Texts cannot contain empty values."
            )

        result = await self._provider_manager.embed(
            list(texts),
            provider=settings.ai.embedding_provider,
        )

        return result.embeddings


embeddings = EmbeddingManager()