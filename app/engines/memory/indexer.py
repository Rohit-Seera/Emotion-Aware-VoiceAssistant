"""Qdrant indexing for persisted memories."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from qdrant_client.models import PointStruct

from app.engines.memory.models import Memory
from app.infrastructure.vectorstore.client import qdrant_client
from app.infrastructure.vectorstore.collections import collections
from app.infrastructure.vectorstore.embeddings import embeddings


class MemoryIndexer:
    """Index memories in Qdrant for semantic retrieval."""

    async def index(self, memory: Memory) -> None:
        """Create an embedding and upsert the memory into Qdrant."""

        if memory.id is None:
            raise ValueError("Memory must have an ID before indexing.")

        vector = await embeddings.embed(memory.content)

        await collections.create()

        payload = {
            "id": str(memory.id),
            "user_id": str(memory.user_id),
            "character_id": (
                str(memory.character_id)
                if memory.character_id is not None
                else None
            ),
            "content": memory.content,
            "memory_type": str(memory.memory_type),
            "source": str(memory.source),
            "status": str(memory.status),
            "importance": float(memory.importance),
            "metadata": memory.metadata_ or {},
            "created_at": self._serialize_datetime(
                memory.created_at,
            ),
            "updated_at": self._serialize_datetime(
                memory.updated_at,
            ),
        }

        await qdrant_client.upsert(
            collection_name=collections.name,
            points=[
                PointStruct(
                    id=str(memory.id),
                    vector=vector,
                    payload=payload,
                ),
            ],
        )

    async def delete(self, memory_id: UUID) -> None:
        """Remove a memory from Qdrant."""

        await qdrant_client.delete(
            collection_name=collections.name,
            points_selector=[
                str(memory_id),
            ],
        )

    @staticmethod
    def _serialize_datetime(
        value: datetime | None,
    ) -> str | None:
        """Serialize a datetime for Qdrant payload storage."""

        if value is None:
            return None

        return value.isoformat()


memory_indexer = MemoryIndexer()