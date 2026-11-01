"""Tests for the DeepSeek provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.providers.base.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.deepseek import DeepSeekProvider


@pytest.fixture
def deepseek_provider(
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> DeepSeekProvider:
    """Return a DeepSeek provider with fully isolated test settings."""

    config = __import__(
        "app.providers.deepseek.provider",
        fromlist=["settings"],
    ).settings.ai.deepseek

    monkeypatch.setattr(config, "api_key", "test-api-key")
    monkeypatch.setattr(config, "enabled", True)
    monkeypatch.setattr(config, "model", "deepseek-chat")
    monkeypatch.setattr(config, "temperature", 0.7)
    monkeypatch.setattr(config, "max_tokens", 4096)

    monkeypatch.setattr(
        "app.providers.deepseek.provider.AsyncOpenAI",
        lambda **kwargs: mock_async_client,
    )

    return DeepSeekProvider()



@pytest.fixture
def mock_deepseek_response() -> MagicMock:
    """Return a mocked DeepSeek chat completion response."""

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
    response.model = "deepseek-test-model"
    response.choices = [choice]
    response._request_id = "test-request-id"

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    return response


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """DeepSeek provider should expose the correct name."""

    provider = DeepSeekProvider.__new__(
        DeepSeekProvider,
    )

    assert provider.name == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_usage() -> None:
    """DeepSeek usage should be converted correctly."""

    response = MagicMock()

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    result = DeepSeekProvider._usage(response)

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

    result = DeepSeekProvider._usage(response)

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

    result = DeepSeekProvider._usage(response)

    assert result is not None

    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
    mock_deepseek_response: MagicMock,
) -> None:
    """DeepSeek should generate a complete response."""

    create = AsyncMock(
        return_value=mock_deepseek_response,
    )

    deepseek_provider._client.chat.completions.create = (
        create
    )

    response = await deepseek_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "deepseek"

    assert response.model == (
        "deepseek-test-model"
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
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
    mock_deepseek_response: MagicMock,
) -> None:
    """DeepSeek should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_deepseek_response,
    )

    deepseek_provider._client.chat.completions.create = (
        create
    )

    await deepseek_provider.generate(
        sample_messages,
        model="deepseek-test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == (
        "deepseek-test-model"
    )

    assert request["messages"] == (
        sample_messages
    )

    assert request["temperature"] == 0.2

    assert request["max_tokens"] == 512


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_uses_config_defaults(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
    mock_deepseek_response: MagicMock,
) -> None:
    """DeepSeek should use configured defaults."""

    create = AsyncMock(
        return_value=mock_deepseek_response,
    )

    deepseek_provider._client.chat.completions.create = (
        create
    )

    await deepseek_provider.generate(
        sample_messages,
    )

    request = create.await_args.kwargs

    assert request["model"] == "deepseek-chat"

    assert request["temperature"] == 0.7

    assert request["max_tokens"] == 4096


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_extra_kwargs(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
    mock_deepseek_response: MagicMock,
) -> None:
    """DeepSeek should forward additional arguments."""

    create = AsyncMock(
        return_value=mock_deepseek_response,
    )

    deepseek_provider._client.chat.completions.create = (
        create
    )

    await deepseek_provider.generate(
        sample_messages,
        top_p=0.9,
    )

    request = create.await_args.kwargs

    assert request["top_p"] == 0.9


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_no_choices(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """DeepSeek should reject responses without choices."""

    response = MagicMock()
    response.choices = []

    deepseek_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no choices",
    ):
        await deepseek_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_content(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """DeepSeek should handle empty message content."""

    message = MagicMock()
    message.content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "deepseek-test-model"
    response.id = "test-response-id"
    response._request_id = "test-request-id"
    response.usage = None

    deepseek_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    result = await deepseek_provider.generate(
        sample_messages,
    )

    assert result.content == ""

    assert result.provider == "deepseek"

    assert result.metadata == {
        "response_id": "test-response-id",
    }


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """DeepSeek should stream response chunks."""

    first_delta = MagicMock()
    first_delta.content = "Hello"

    first_choice = MagicMock()
    first_choice.delta = first_delta
    first_choice.finish_reason = None

    first_chunk = MagicMock()
    first_chunk.choices = [first_choice]
    first_chunk.model = "deepseek-test-model"
    first_chunk._request_id = "test-request-id"

    second_delta = MagicMock()
    second_delta.content = " world"

    second_choice = MagicMock()
    second_choice.delta = second_delta
    second_choice.finish_reason = None

    second_chunk = MagicMock()
    second_chunk.choices = [second_choice]
    second_chunk.model = "deepseek-test-model"
    second_chunk._request_id = "test-request-id"

    final_delta = MagicMock()
    final_delta.content = None

    final_choice = MagicMock()
    final_choice.delta = final_delta
    final_choice.finish_reason = "stop"

    final_chunk = MagicMock()
    final_chunk.choices = [final_choice]
    final_chunk.model = "deepseek-test-model"

    async def fake_stream():
        yield first_chunk
        yield second_chunk
        yield final_chunk

    deepseek_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in deepseek_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 2

    assert chunks[0].content == "Hello"
    assert chunks[0].provider == "deepseek"

    assert chunks[1].content == " world"

    request = (
        deepseek_provider
        ._client
        .chat
        .completions
        .create
        .await_args.kwargs
    )

    assert request["stream"] is True


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_empty_chunks(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """DeepSeek should skip chunks without content."""

    empty_delta = MagicMock()
    empty_delta.content = None

    empty_choice = MagicMock()
    empty_choice.delta = empty_delta
    empty_choice.finish_reason = None

    empty_chunk = MagicMock()
    empty_chunk.choices = [empty_choice]
    empty_chunk.model = "deepseek-test-model"

    valid_delta = MagicMock()
    valid_delta.content = "Hello"

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "deepseek-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    deepseek_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in deepseek_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1

    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_chunks_without_choices(
    deepseek_provider: DeepSeekProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """DeepSeek should skip chunks without choices."""

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    valid_delta = MagicMock()
    valid_delta.content = "Hello"

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "deepseek-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    deepseek_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in deepseek_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1

    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_not_supported(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """DeepSeek should reject embedding requests."""

    with pytest.raises(
        ProviderRequestError,
        match="does not provide an embeddings API",
    ):
        await deepseek_provider.embed(
            ["Hello"],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """DeepSeek health check should return True."""

    deepseek_provider._client.models.retrieve = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await deepseek_provider.health_check()

    assert result is True

    deepseek_provider._client.models.retrieve.assert_awaited_once_with(
        "deepseek-chat",
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """DeepSeek health check should return False."""

    deepseek_provider._client.models.retrieve = (
        AsyncMock(
            side_effect=Exception(
                "DeepSeek unavailable",
            ),
        )
    )

    result = await deepseek_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    deepseek_provider: DeepSeekProvider,
) -> None:
    """DeepSeek client should close cleanly."""

    deepseek_provider._client.close = AsyncMock()

    await deepseek_provider.close()

    deepseek_provider._client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """DeepSeek authentication errors should be normalized."""

    error = openai.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """DeepSeek rate-limit errors should be normalized."""

    error = openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """DeepSeek timeout errors should be normalized."""

    error = openai.APITimeoutError(
        request=MagicMock(),
    )

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """DeepSeek connection errors should be normalized."""

    error = openai.APIConnectionError(
        request=MagicMock(),
    )

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_server_error() -> None:
    """DeepSeek 5xx errors should be unavailable errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 500

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_client_error() -> None:
    """DeepSeek 4xx errors should be request errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 400

    result = DeepSeekProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "deepseek"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = DeepSeekProvider._map_error(
        RuntimeError("Unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "deepseek"