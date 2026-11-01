"""Qdrant collection management."""

from __future__ import annotations

from qdrant_client.http.models import Distance
from qdrant_client.http.models import VectorParams

from app.core.settings import settings
from app.infrastructure.vectorstore.client import qdrant_client


class CollectionManager:
    """Manage Qdrant collections."""

    @property
    def name(self) -> str:
        """Return the default collection name."""

        return settings.qdrant.collection

    async def exists(self) -> bool:
        """Check whether the collection exists."""

        return await qdrant_client.collection_exists(
            self.name,
        )

    async def create(self) -> None:
        """Create the default collection."""

        if await self.exists():
            return

        await qdrant_client.create_collection(
            collection_name=self.name,
            vectors_config=VectorParams(
                size=settings.memory.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )

    async def delete(self) -> None:
        """Delete the default collection."""

        if not await self.exists():
            return

        await qdrant_client.delete_collection(
            collection_name=self.name,
        )


collections = CollectionManager()