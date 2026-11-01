"""Tests for the OpenRouter provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from app.core.settings import settings
from app.providers.base.exceptions import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.openrouter import OpenRouterProvider


@pytest.fixture
def openrouter_provider(
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> OpenRouterProvider:
    """Return an OpenRouter provider with isolated test settings."""

    config = settings.ai.openrouter

    monkeypatch.setattr(
        config,
        "api_key",
        "test-api-key",
    )
    monkeypatch.setattr(
        config,
        "enabled",
        True,
    )
    monkeypatch.setattr(
        config,
        "base_url",
        "https://openrouter.ai/api/v1",
    )
    monkeypatch.setattr(
        config,
        "temperature",
        0.7,
    )
    monkeypatch.setattr(
        config,
        "max_tokens",
        4096,
    )

    return OpenRouterProvider()


@pytest.fixture
def mock_openrouter_response() -> MagicMock:
    """Return a mocked OpenRouter chat completion response."""

    message = MagicMock()
    message.content = (
        "Hello! I am working correctly."
    )
    message.role = "assistant"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.id = "test-response-id"
    response.model = "openrouter-test-model"
    response.choices = [choice]
    response._request_id = "test-request-id"

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    return response


@pytest.fixture
def mock_openrouter_embedding_response() -> MagicMock:
    """Return a mocked OpenRouter embedding response."""

    first = MagicMock()
    first.index = 0
    first.embedding = [0.1, 0.2, 0.3]

    second = MagicMock()
    second.index = 1
    second.embedding = [0.4, 0.5, 0.6]

    response = MagicMock()
    response.model = "openrouter-embedding-model"
    response.data = [first, second]

    response.usage = MagicMock(
        prompt_tokens=12,
        completion_tokens=0,
        total_tokens=12,
    )

    return response


# ---------------------------------------------------------------------------
# Basic / usage
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """OpenRouter provider should expose the correct name."""

    provider = OpenRouterProvider.__new__(
        OpenRouterProvider,
    )

    assert provider.name == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_usage() -> None:
    """OpenRouter usage should be converted correctly."""

    response = MagicMock()

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    result = OpenRouterProvider._usage(response)

    assert result is not None
    assert result.prompt_tokens == 20
    assert result.completion_tokens == 10
    assert result.total_tokens == 30


@pytest.mark.unit
@pytest.mark.provider
def test_usage_without_metadata() -> None:
    """Missing usage metadata should return None."""

    response = MagicMock()
    response.usage = None

    result = OpenRouterProvider._usage(response)

    assert result is None


@pytest.mark.unit
@pytest.mark.provider
def test_usage_missing_values() -> None:
    """Missing usage values should default to zero."""

    response = MagicMock()

    response.usage = MagicMock(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    result = OpenRouterProvider._usage(response)

    assert result is not None
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
    mock_openrouter_response: MagicMock,
) -> None:
    """OpenRouter should generate a complete response."""

    create = AsyncMock(
        return_value=mock_openrouter_response,
    )

    openrouter_provider._client.chat.completions.create = (
        create
    )

    response = await openrouter_provider.generate(
        sample_messages,
        model="openrouter-test-model",
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "openrouter"
    assert response.model == (
        "openrouter-test-model"
    )
    assert response.finish_reason == "stop"
    assert response.request_id == (
        "test-request-id"
    )

    assert response.metadata == {
        "response_id": "test-response-id",
    }

    assert response.usage is not None
    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 30

    create.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_parameters(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
    mock_openrouter_response: MagicMock,
) -> None:
    """OpenRouter should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_openrouter_response,
    )

    openrouter_provider._client.chat.completions.create = (
        create
    )

    await openrouter_provider.generate(
        sample_messages,
        model="openrouter-test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == (
        "openrouter-test-model"
    )
    assert request["messages"] == sample_messages
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 512


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_uses_config_defaults(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
    mock_openrouter_response: MagicMock,
) -> None:
    """OpenRouter should use configured generation defaults."""

    create = AsyncMock(
        return_value=mock_openrouter_response,
    )

    openrouter_provider._client.chat.completions.create = (
        create
    )

    await openrouter_provider.generate(
        sample_messages,
        model="openrouter-test-model",
    )

    request = create.await_args.kwargs

    assert request["model"] == (
        "openrouter-test-model"
    )
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 4096


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_extra_kwargs(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
    mock_openrouter_response: MagicMock,
) -> None:
    """OpenRouter should forward additional arguments."""

    create = AsyncMock(
        return_value=mock_openrouter_response,
    )

    openrouter_provider._client.chat.completions.create = (
        create
    )

    await openrouter_provider.generate(
        sample_messages,
        model="openrouter-test-model",
        top_p=0.9,
    )

    request = create.await_args.kwargs

    assert request["top_p"] == 0.9


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_requires_model(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should require an explicit model."""

    with pytest.raises(
        ProviderConfigurationError,
        match="requires an explicit model",
    ):
        await openrouter_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_no_choices(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should reject responses without choices."""

    response = MagicMock()
    response.choices = []

    openrouter_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no choices",
    ):
        await openrouter_provider.generate(
            sample_messages,
            model="openrouter-test-model",
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_content(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should handle empty message content."""

    message = MagicMock()
    message.content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "openrouter-test-model"
    response.id = "test-response-id"
    response._request_id = "test-request-id"
    response.usage = None

    openrouter_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    result = await openrouter_provider.generate(
        sample_messages,
        model="openrouter-test-model",
    )

    assert result.content == ""

    assert result.metadata == {
        "response_id": "test-response-id",
    }


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should stream response chunks."""

    delta = MagicMock()
    delta.content = "Hello"

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = "openrouter-test-model"
    chunk._request_id = "test-request-id"

    async def fake_stream():
        yield chunk

    create = AsyncMock(
        return_value=fake_stream(),
    )

    openrouter_provider._client.chat.completions.create = (
        create
    )

    chunks = [
        chunk
        async for chunk in openrouter_provider.stream(
            sample_messages,
            model="openrouter-test-model",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"
    assert chunks[0].provider == "openrouter"

    request = create.await_args.kwargs

    assert request["model"] == (
        "openrouter-test-model"
    )
    assert request["stream"] is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_requires_model(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter streaming should require an explicit model."""

    with pytest.raises(
        ProviderConfigurationError,
        match="requires an explicit model",
    ):
        async for _ in openrouter_provider.stream(
            sample_messages,
        ):
            pass


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_empty_chunks(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should skip chunks without content."""

    empty_delta = MagicMock()
    empty_delta.content = None

    empty_choice = MagicMock()
    empty_choice.delta = empty_delta
    empty_choice.finish_reason = None

    empty_chunk = MagicMock()
    empty_chunk.choices = [empty_choice]

    valid_delta = MagicMock()
    valid_delta.content = "Hello"

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "openrouter-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    openrouter_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in openrouter_provider.stream(
            sample_messages,
            model="openrouter-test-model",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_chunks_without_choices(
    openrouter_provider: OpenRouterProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """OpenRouter should skip chunks without choices."""

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    delta = MagicMock()
    delta.content = "Hello"

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [choice]
    valid_chunk.model = "openrouter-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    openrouter_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in openrouter_provider.stream(
            sample_messages,
            model="openrouter-test-model",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_empty_input(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """OpenRouter should return an empty result for empty input."""

    result = await openrouter_provider.embed(
        [],
        model="openrouter-embedding-model",
    )

    assert result.embeddings == []
    assert result.provider == "openrouter"
    assert result.model == (
        "openrouter-embedding-model"
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_requires_model(
    openrouter_provider: OpenRouterProvider,
    sample_embedding_texts: list[str],
) -> None:
    """OpenRouter embeddings should require an explicit model."""

    with pytest.raises(
        ProviderConfigurationError,
        match="embedding requires an explicit model",
    ):
        await openrouter_provider.embed(
            sample_embedding_texts,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed(
    openrouter_provider: OpenRouterProvider,
    sample_embedding_texts: list[str],
    mock_openrouter_embedding_response: MagicMock,
) -> None:
    """OpenRouter should generate embeddings."""

    create = AsyncMock(
        return_value=mock_openrouter_embedding_response,
    )

    openrouter_provider._client.embeddings.create = (
        create
    )

    result = await openrouter_provider.embed(
        sample_embedding_texts,
        model="openrouter-embedding-model",
    )

    assert result.provider == "openrouter"
    assert result.model == (
        "openrouter-embedding-model"
    )

    assert result.embeddings == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    assert result.usage is not None
    assert result.usage.prompt_tokens == 12
    assert result.usage.total_tokens == 12

    request = create.await_args.kwargs

    assert request["model"] == (
        "openrouter-embedding-model"
    )
    assert request["input"] == (
        sample_embedding_texts
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_passes_extra_kwargs(
    openrouter_provider: OpenRouterProvider,
    mock_openrouter_embedding_response: MagicMock,
) -> None:
    """OpenRouter should forward embedding kwargs."""

    create = AsyncMock(
        return_value=mock_openrouter_embedding_response,
    )

    openrouter_provider._client.embeddings.create = (
        create
    )

    await openrouter_provider.embed(
        ["Hello"],
        model="openrouter-embedding-model",
        dimensions=256,
    )

    request = create.await_args.kwargs

    assert request["dimensions"] == 256


# ---------------------------------------------------------------------------
# Health / lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """OpenRouter health check should return True."""

    openrouter_provider._client.models.list = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await openrouter_provider.health_check()

    assert result is True

    openrouter_provider._client.models.list.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """OpenRouter health check should return False."""

    openrouter_provider._client.models.list = (
        AsyncMock(
            side_effect=Exception(
                "OpenRouter unavailable",
            ),
        )
    )

    result = await openrouter_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    openrouter_provider: OpenRouterProvider,
) -> None:
    """OpenRouter client should close cleanly."""

    openrouter_provider._client.close = AsyncMock()

    await openrouter_provider.close()

    openrouter_provider._client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """OpenRouter authentication errors should be normalized."""

    error = openai.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """OpenRouter rate-limit errors should be normalized."""

    error = openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """OpenRouter timeout errors should be normalized."""

    error = openai.APITimeoutError(
        request=MagicMock(),
    )

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """OpenRouter connection errors should be normalized."""

    error = openai.APIConnectionError(
        request=MagicMock(),
    )

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_server_error() -> None:
    """OpenRouter 5xx errors should be unavailable errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 500

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_client_error() -> None:
    """OpenRouter 4xx errors should be request errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 400

    result = OpenRouterProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "openrouter"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = OpenRouterProvider._map_error(
        RuntimeError("Unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "openrouter"