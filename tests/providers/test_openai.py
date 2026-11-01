"""Tests for the OpenAI provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.providers.base.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from app.providers.openai import OpenAIProvider


@pytest.fixture
def openai_provider(
    mock_async_client: MagicMock,
) -> OpenAIProvider:
    """Return an OpenAI provider with a mocked client."""

    with (
        patch(
            "app.providers.openai.provider.AsyncOpenAI",
            return_value=mock_async_client,
        ),
        patch(
            "app.providers.openai.provider.settings.ai.openai.api_key",
            "test-api-key",
        ),
    ):
        provider = OpenAIProvider()

    return provider


@pytest.fixture
def mock_response() -> MagicMock:
    """Return a mocked OpenAI Responses API response."""

    response = MagicMock()

    response.id = "test-response-id"
    response.model = "test-model"
    response.output_text = (
        "Hello! I am working correctly."
    )
    response._request_id = "test-request-id"

    response.status = "completed"

    response.usage = MagicMock(
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
    )

    return response


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """OpenAI provider should expose the correct name."""

    provider = OpenAIProvider.__new__(
        OpenAIProvider,
    )

    assert provider.name == "openai"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
    mock_response: MagicMock,
) -> None:
    """OpenAI should generate a complete response."""

    create = AsyncMock(
        return_value=mock_response,
    )

    openai_provider._client.responses.create = create

    response = await openai_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "openai"

    assert response.model == "test-model"

    assert response.request_id == (
        "test-request-id"
    )

    create.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_parameters(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
    mock_response: MagicMock,
) -> None:
    """OpenAI should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_response,
    )

    openai_provider._client.responses.create = create

    await openai_provider.generate(
        sample_messages,
        model="test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == "test-model"

    assert request["input"] == sample_messages

    assert request["temperature"] == 0.2

    assert request["max_output_tokens"] == 512


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_authentication_error(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenAI authentication errors should be normalized."""

    error = openai.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    openai_provider._client.responses.create = (
        AsyncMock(
            side_effect=error,
        )
    )

    with pytest.raises(
        ProviderAuthenticationError,
    ):
        await openai_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_rate_limit_error(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenAI rate-limit errors should be normalized."""

    error = openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    openai_provider._client.responses.create = (
        AsyncMock(
            side_effect=error,
        )
    )

    with pytest.raises(
        ProviderRateLimitError,
    ):
        await openai_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_timeout_error(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenAI timeout errors should be normalized."""

    error = openai.APITimeoutError(
        request=MagicMock(),
    )

    openai_provider._client.responses.create = (
        AsyncMock(
            side_effect=error,
        )
    )

    with pytest.raises(
        ProviderTimeoutError,
    ):
        await openai_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    openai_provider: OpenAIProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenAI should stream Responses API events."""

    delta_event = MagicMock()

    delta_event.type = (
        "response.output_text.delta"
    )
    delta_event.delta = "Hello"

    done_event = MagicMock()

    done_event.type = (
        "response.output_text.done"
    )

    completed_response = MagicMock()

    completed_response.model = "test-model"
    completed_response.id = (
        "test-response-id"
    )
    completed_response._request_id = (
        "test-request-id"
    )

    completed_event = MagicMock()

    completed_event.type = (
        "response.completed"
    )
    completed_event.response = (
        completed_response
    )

    async def fake_stream():
        yield delta_event
        yield done_event
        yield completed_event

    create = AsyncMock(
        return_value=fake_stream(),
    )

    openai_provider._client.responses.create = create

    chunks = [
        chunk
        async for chunk in openai_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 3

    assert chunks[0].content == "Hello"

    assert chunks[0].provider == "openai"

    assert chunks[1].content == ""

    assert chunks[1].metadata == {
        "text_completed": True,
    }

    assert chunks[2].finish_reason == "completed"

    assert chunks[2].request_id == (
        "test-request-id"
    )

    request = create.await_args.kwargs

    assert request["stream"] is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed(
    openai_provider: OpenAIProvider,
    sample_embedding_texts: list[str],
    mock_embedding_response: MagicMock,
) -> None:
    """OpenAI should generate embeddings."""

    openai_provider._client.embeddings.create = (
        AsyncMock(
            return_value=mock_embedding_response,
        )
    )

    result = await openai_provider.embed(
        sample_embedding_texts,
    )

    assert result.provider == "openai"

    assert result.model == (
        "test-embedding-model"
    )

    assert result.embeddings == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    (
        openai_provider
        ._client
        .embeddings
        .create
        .assert_awaited_once()
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    openai_provider: OpenAIProvider,
) -> None:
    """Health check should return True when API responds."""

    openai_provider._client.models.retrieve = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await openai_provider.health_check()

    assert result is True

    (
        openai_provider
        ._client
        .models
        .retrieve
        .assert_awaited_once()
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    openai_provider: OpenAIProvider,
) -> None:
    """Health check should return False on failure."""

    openai_provider._client.models.retrieve = (
        AsyncMock(
            side_effect=ProviderError(
                "Provider unavailable.",
                provider="openai",
            ),
        )
    )

    result = await openai_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    openai_provider: OpenAIProvider,
) -> None:
    """OpenAI client should close cleanly."""

    openai_provider._client.close = AsyncMock()

    await openai_provider.close()

    openai_provider._client.close.assert_awaited_once()