"""Qdrant client."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from app.core.settings import settings


def create_qdrant_client() -> AsyncQdrantClient:
    """Create the Qdrant client."""

    return AsyncQdrantClient(
        url=settings.qdrant.url,
        api_key=settings.qdrant.api_key,
        prefer_grpc=settings.qdrant.prefer_grpc,
    )


qdrant_client = create_qdrant_client()