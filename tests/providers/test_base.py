"""Tests for the base provider contract."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from app.providers.base.exceptions import ProviderError
from app.providers.base.provider import BaseProvider
from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
)


class FakeProvider(BaseProvider):
    """Concrete provider used to test the base provider contract."""

    name = "fake"

    def __init__(
        self,
        *,
        generate_response: ProviderResponse | None = None,
        embedding_result: EmbeddingResult | None = None,
        stream_chunks: list[ProviderChunk] | None = None,
        healthy: bool = True,
    ) -> None:
        self.generate_response = generate_response
        self.embedding_result = embedding_result
        self.stream_chunks = stream_chunks or []
        self.healthy = healthy

        self.generate_called = False
        self.embed_called = False
        self.stream_called = False
        self.close_called = False

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Return the configured fake response."""

        self.generate_called = True

        if self.generate_response is None:
            raise ProviderError(
                "Fake generation failed.",
                provider=self.name,
            )

        return self.generate_response

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderChunk]:
        """Yield configured fake streaming chunks."""

        self.stream_called = True

        for chunk in self.stream_chunks:
            yield chunk

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Return the configured fake embedding result."""

        self.embed_called = True

        if self.embedding_result is None:
            raise ProviderError(
                "Fake embedding failed.",
                provider=self.name,
            )

        return self.embedding_result

    async def health_check(self) -> bool:
        """Return the configured fake health status."""

        return self.healthy

    async def close(self) -> None:
        """Mark the fake provider as closed."""

        self.close_called = True


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """Provider should expose a stable provider name."""

    provider = FakeProvider()

    assert provider.name == "fake"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    sample_messages: list[dict[str, str]],
    sample_response: ProviderResponse,
) -> None:
    """Provider should generate a response."""

    provider = FakeProvider(
        generate_response=sample_response,
    )

    response = await provider.generate(
        sample_messages,
    )

    assert response is sample_response
    assert response.content == (
        "Hello! I am working correctly."
    )
    assert provider.generate_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_raises_provider_error(
    sample_messages: list[dict[str, str]],
) -> None:
    """Provider should expose generation failures."""

    provider = FakeProvider()

    with pytest.raises(
        ProviderError,
        match="Fake generation failed",
    ):
        await provider.generate(
            sample_messages,
        )

    assert provider.generate_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    sample_messages: list[dict[str, str]],
    sample_chunk: ProviderChunk,
    sample_final_chunk: ProviderChunk,
) -> None:
    """Provider should stream chunks in order."""

    provider = FakeProvider(
        stream_chunks=[
            sample_chunk,
            sample_final_chunk,
        ],
    )

    chunks = [
        chunk
        async for chunk in provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 2
    assert chunks[0] is sample_chunk
    assert chunks[1] is sample_final_chunk

    assert provider.stream_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_empty_result(
    sample_messages: list[dict[str, str]],
) -> None:
    """Provider should allow an empty stream."""

    provider = FakeProvider()

    chunks = [
        chunk
        async for chunk in provider.stream(
            sample_messages,
        )
    ]

    assert chunks == []
    assert provider.stream_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed(
    sample_embedding_texts: list[str],
    sample_embedding_result: EmbeddingResult,
) -> None:
    """Provider should generate embeddings."""

    provider = FakeProvider(
        embedding_result=sample_embedding_result,
    )

    result = await provider.embed(
        sample_embedding_texts,
    )

    assert result is sample_embedding_result
    assert len(result.embeddings) == 2
    assert provider.embed_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_raises_provider_error(
    sample_embedding_texts: list[str],
) -> None:
    """Provider should expose embedding failures."""

    provider = FakeProvider()

    with pytest.raises(
        ProviderError,
        match="Fake embedding failed",
    ):
        await provider.embed(
            sample_embedding_texts,
        )

    assert provider.embed_called is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_success() -> None:
    """Healthy provider should return True."""

    provider = FakeProvider(
        healthy=True,
    )

    result = await provider.health_check()

    assert result is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure() -> None:
    """Unhealthy provider should return False."""

    provider = FakeProvider(
        healthy=False,
    )

    result = await provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close() -> None:
    """Provider should support clean shutdown."""

    provider = FakeProvider()

    await provider.close()

    assert provider.close_called is True