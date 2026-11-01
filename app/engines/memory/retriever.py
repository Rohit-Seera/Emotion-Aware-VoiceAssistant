"""Semantic memory retrieval."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.engines.memory.schemas import (
    MemoryRead,
    MemorySearchResult,
)
from app.engines.memory.types import (
    MemorySource,
    MemoryStatus,
    MemoryType,
)
from app.infrastructure.vectorstore.client import qdrant_client
from app.infrastructure.vectorstore.collections import collections
from app.infrastructure.vectorstore.embeddings import embeddings


class MemoryRetriever:
    """Retrieve semantically relevant memories from Qdrant."""

    async def retrieve(
        self,
        query: str,
        *,
        user_id: UUID,
        character_id: UUID | None = None,
        top_k: int = 10,
        similarity_threshold: float = 0.75,
    ) -> list[MemorySearchResult]:
        """Retrieve memories scoped to a user and optionally a character."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            return []

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "Similarity threshold must be between 0 and 1."
            )

        vector = await embeddings.embed(query)

        conditions = [
            FieldCondition(
                key="user_id",
                match=MatchValue(
                    value=str(user_id),
                ),
            ),
        ]

        if character_id is not None:
            conditions.append(
                FieldCondition(
                    key="character_id",
                    match=MatchValue(
                        value=str(character_id),
                    ),
                )
            )

        memory_filter = Filter(
            must=conditions,
        )

        response = await qdrant_client.query_points(
            collection_name=collections.name,
            query=vector,
            query_filter=memory_filter,
            limit=top_k,
            score_threshold=similarity_threshold,
        )

        memories: list[MemorySearchResult] = []

        for result in response.points:
            payload = result.payload or {}

            memory = MemoryRead(
                id=UUID(str(payload["id"])),
                user_id=UUID(str(payload["user_id"])),
                character_id=(
                    UUID(str(payload["character_id"]))
                    if payload.get("character_id") is not None
                    else None
                ),
                content=str(payload["content"]),
                memory_type=MemoryType(
                    payload["memory_type"],
                ),
                source=MemorySource(
                    payload["source"],
                ),
                status=MemoryStatus(
                    payload["status"],
                ),
                importance=float(
                    payload["importance"],
                ),
                metadata=payload.get(
                    "metadata",
                    {},
                ),
                created_at=(
                    payload["created_at"]
                    if isinstance(
                        payload["created_at"],
                        datetime,
                    )
                    else datetime.fromisoformat(
                        str(payload["created_at"]).replace(
                            "Z",
                            "+00:00",
                        )
                    )
                ),
                updated_at=(
                    payload["updated_at"]
                    if isinstance(
                        payload["updated_at"],
                        datetime,
                    )
                    else datetime.fromisoformat(
                        str(payload["updated_at"]).replace(
                            "Z",
                            "+00:00",
                        )
                    )
                ),
            )

            memories.append(
                MemorySearchResult(
                    memory=memory,
                    score=float(result.score),
                )
            )

        return memories