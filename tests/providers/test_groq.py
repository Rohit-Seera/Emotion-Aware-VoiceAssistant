"""Tests for the Groq provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import groq
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
from app.providers.groq import GroqProvider


@pytest.fixture
def groq_provider(
    mock_groq_client: MagicMock,
) -> GroqProvider:
    """Return a Groq provider with a mocked client."""

    with (
        patch(
            "app.providers.groq.provider.AsyncGroq",
            return_value=mock_groq_client,
        ),
        patch(
            "app.providers.groq.provider.settings.ai.groq.api_key",
            "test-api-key",
        ),
    ):
        provider = GroqProvider()

    return provider


@pytest.fixture
def mock_groq_response() -> MagicMock:
    """Return a mocked Groq chat completion response."""

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
    response.model = "llama-test-model"
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
    """Groq provider should expose the correct name."""

    provider = GroqProvider.__new__(
        GroqProvider,
    )

    assert provider.name == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_usage() -> None:
    """Groq usage should be converted correctly."""

    response = MagicMock()

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    usage = GroqProvider._usage(response)

    assert usage is not None

    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 30


@pytest.mark.unit
@pytest.mark.provider
def test_usage_without_metadata() -> None:
    """Missing usage metadata should return None."""

    response = MagicMock()
    response.usage = None

    result = GroqProvider._usage(response)

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

    result = GroqProvider._usage(response)

    assert result is not None

    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
    mock_groq_response: MagicMock,
) -> None:
    """Groq should generate a complete response."""

    create = AsyncMock(
        return_value=mock_groq_response,
    )

    groq_provider._client.chat.completions.create = (
        create
    )

    response = await groq_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "groq"

    assert response.model == "llama-test-model"

    assert response.finish_reason == "stop"

    assert response.request_id == (
        "test-request-id"
    )

    assert response.usage is not None

    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 30

    create.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_parameters(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
    mock_groq_response: MagicMock,
) -> None:
    """Groq should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_groq_response,
    )

    groq_provider._client.chat.completions.create = (
        create
    )

    await groq_provider.generate(
        sample_messages,
        model="llama-test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == "llama-test-model"

    assert request["messages"] == (
        sample_messages
    )

    assert request["temperature"] == 0.2

    assert request["max_tokens"] == 512


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_uses_config_defaults(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
    mock_groq_response: MagicMock,
) -> None:
    """Groq should use configured defaults."""

    create = AsyncMock(
        return_value=mock_groq_response,
    )

    groq_provider._client.chat.completions.create = (
        create
    )

    await groq_provider.generate(
        sample_messages,
    )

    request = create.await_args.kwargs

    assert request["model"] == (
        "llama-3.3-70b-versatile"
    )

    assert request["temperature"] == 0.7

    assert request["max_tokens"] == 4096


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_extra_kwargs(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
    mock_groq_response: MagicMock,
) -> None:
    """Groq should forward additional request arguments."""

    create = AsyncMock(
        return_value=mock_groq_response,
    )

    groq_provider._client.chat.completions.create = (
        create
    )

    await groq_provider.generate(
        sample_messages,
        top_p=0.9,
        stream_options={
            "include_usage": True,
        },
    )

    request = create.await_args.kwargs

    assert request["top_p"] == 0.9

    assert request["stream_options"] == {
        "include_usage": True,
    }


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_no_choices(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Groq should reject responses without choices."""

    response = MagicMock()
    response.choices = []

    groq_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no choices",
    ):
        await groq_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_content(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Groq should handle empty message content."""

    message = MagicMock()
    message.content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "llama-test-model"
    response._request_id = "test-request-id"
    response.usage = None

    groq_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    result = await groq_provider.generate(
        sample_messages,
    )

    assert result.content == ""

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Groq should stream response chunks."""

    first_delta = MagicMock()
    first_delta.content = "Hello"

    first_choice = MagicMock()
    first_choice.delta = first_delta
    first_choice.finish_reason = None

    first_chunk = MagicMock()
    first_chunk.choices = [first_choice]
    first_chunk.model = "llama-test-model"
    first_chunk._request_id = "test-request-id"

    second_delta = MagicMock()
    second_delta.content = " world"

    second_choice = MagicMock()
    second_choice.delta = second_delta
    second_choice.finish_reason = None

    second_chunk = MagicMock()
    second_chunk.choices = [second_choice]
    second_chunk.model = "llama-test-model"
    second_chunk._request_id = "test-request-id"

    final_delta = MagicMock()
    final_delta.content = None

    final_choice = MagicMock()
    final_choice.delta = final_delta
    final_choice.finish_reason = "stop"

    final_chunk = MagicMock()
    final_chunk.choices = [final_choice]
    final_chunk.model = "llama-test-model"
    final_chunk._request_id = "test-request-id"

    async def fake_stream():
        yield first_chunk
        yield second_chunk
        yield final_chunk

    groq_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in groq_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 2

    assert chunks[0].content == "Hello"
    assert chunks[0].provider == "groq"

    assert chunks[1].content == " world"

    groq_provider._client.chat.completions.create.assert_awaited_once()

    request = (
        groq_provider
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
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Groq should skip chunks without content."""

    empty_delta = MagicMock()
    empty_delta.content = None

    empty_choice = MagicMock()
    empty_choice.delta = empty_delta
    empty_choice.finish_reason = None

    empty_chunk = MagicMock()
    empty_chunk.choices = [empty_choice]
    empty_chunk.model = "llama-test-model"

    valid_delta = MagicMock()
    valid_delta.content = "Hello"

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "llama-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    groq_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in groq_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1

    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_chunks_without_choices(
    groq_provider: GroqProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Groq should skip chunks without choices."""

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    valid_delta = MagicMock()
    valid_delta.content = "Hello"

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "llama-test-model"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    groq_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in groq_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1

    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_not_supported(
    groq_provider: GroqProvider,
) -> None:
    """Groq should reject embedding requests."""

    with pytest.raises(
        ProviderRequestError,
        match="does not provide an embedding API",
    ):
        await groq_provider.embed(
            ["Hello"],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    groq_provider: GroqProvider,
) -> None:
    """Groq health check should return True."""

    groq_provider._client.models.retrieve = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await groq_provider.health_check()

    assert result is True

    groq_provider._client.models.retrieve.assert_awaited_once_with(
        "llama-3.3-70b-versatile",
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    groq_provider: GroqProvider,
) -> None:
    """Groq health check should return False."""

    groq_provider._client.models.retrieve = (
        AsyncMock(
            side_effect=Exception(
                "Groq unavailable",
            ),
        )
    )

    result = await groq_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    groq_provider: GroqProvider,
) -> None:
    """Groq client should close cleanly."""

    groq_provider._client.close = AsyncMock()

    await groq_provider.close()

    groq_provider._client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """Groq authentication errors should be normalized."""

    error = groq.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """Groq rate-limit errors should be normalized."""

    error = groq.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """Groq timeout errors should be normalized."""

    error = groq.APITimeoutError(
        request=MagicMock(),
    )

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """Groq connection errors should be normalized."""

    error = groq.APIConnectionError(
        request=MagicMock(),
    )

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_server_error() -> None:
    """Groq 5xx errors should be unavailable errors."""

    error = MagicMock(
        spec=groq.APIStatusError,
    )

    error.status_code = 500

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_client_error() -> None:
    """Groq 4xx errors should be request errors."""

    error = MagicMock(
        spec=groq.APIStatusError,
    )

    error.status_code = 400

    result = GroqProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "groq"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = GroqProvider._map_error(
        RuntimeError("Unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "groq"