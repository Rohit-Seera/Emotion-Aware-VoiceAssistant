"""Tests for the Anthropic provider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from app.providers.anthropic import AnthropicProvider
from app.providers.base.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.fixture
def anthropic_provider(
    mock_anthropic_client: MagicMock,
) -> AnthropicProvider:
    """Return an Anthropic provider with a mocked client."""

    with (
        patch(
            "app.providers.anthropic.provider.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ),
        patch(
            "app.providers.anthropic.provider.settings.ai.anthropic.api_key",
            "test-api-key",
        ),
    ):
        provider = AnthropicProvider()

    return provider


@pytest.fixture
def mock_anthropic_response() -> MagicMock:
    """Return a mocked Anthropic Messages API response."""

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        "Hello! I am working correctly."
    )

    response = MagicMock()
    response.id = "test-response-id"
    response.model = "claude-test-model"
    response.content = [text_block]
    response.stop_reason = "end_turn"
    response.stop_sequence = None
    response._request_id = "test-request-id"

    response.usage = MagicMock(
        input_tokens=20,
        output_tokens=10,
    )

    return response


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """Anthropic provider should expose the correct name."""

    provider = AnthropicProvider.__new__(
        AnthropicProvider,
    )

    assert provider.name == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages() -> None:
    """Anthropic should convert common messages correctly."""

    messages = [
        {
            "role": "system",
            "content": "You are helpful.",
        },
        {
            "role": "user",
            "content": "Hello!",
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
    ]

    prepared, system_instruction = (
        AnthropicProvider._prepare_messages(
            messages,
        )
    )

    assert system_instruction == "You are helpful."

    assert prepared == [
        {
            "role": "user",
            "content": "Hello!",
        },
        {
            "role": "assistant",
            "content": "Hi!",
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_multiple_system_parts() -> None:
    """Anthropic should combine multiple system parts."""

    messages = [
        {
            "role": "system",
            "content": "Be helpful.",
        },
        {
            "role": "system",
            "content": "Be concise.",
        },
    ]

    prepared, system_instruction = (
        AnthropicProvider._prepare_messages(
            messages,
        )
    )

    assert prepared == []

    assert system_instruction == (
        "Be helpful.\n\nBe concise."
    )


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_system_content_parts() -> None:
    """Anthropic should extract text from system content parts."""

    messages = [
        {
            "role": "system",
            "content": [
                {"text": "Be helpful."},
                {"text": "Be concise."},
            ],
        },
    ]

    prepared, system_instruction = (
        AnthropicProvider._prepare_messages(
            messages,
        )
    )

    assert prepared == []

    assert system_instruction == (
        "Be helpful.\n\nBe concise."
    )


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_converts_unknown_role() -> None:
    """Unknown roles should become user messages."""

    messages = [
        {
            "role": "model",
            "content": "Previous answer.",
        },
    ]

    prepared, system_instruction = (
        AnthropicProvider._prepare_messages(
            messages,
        )
    )

    assert system_instruction is None

    assert prepared == [
        {
            "role": "user",
            "content": "Previous answer.",
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_usage() -> None:
    """Anthropic usage should be converted correctly."""

    response = MagicMock()

    response.usage = MagicMock(
        input_tokens=20,
        output_tokens=10,
    )

    usage = AnthropicProvider._usage(
        response,
    )

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

    result = AnthropicProvider._usage(
        response,
    )

    assert result is None


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
    mock_anthropic_response: MagicMock,
) -> None:
    """Anthropic should generate a complete response."""

    create = AsyncMock(
        return_value=mock_anthropic_response,
    )

    anthropic_provider._client.messages.create = (
        create
    )

    response = await anthropic_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "anthropic"

    assert response.model == "claude-test-model"

    assert response.finish_reason == "end_turn"

    assert response.request_id == (
        "test-request-id"
    )

    assert response.usage is not None

    assert response.usage.prompt_tokens == 20

    assert response.usage.completion_tokens == 10

    assert response.usage.total_tokens == 30

    assert response.metadata == {
        "response_id": "test-response-id",
        "stop_sequence": None,
    }

    create.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_parameters(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
    mock_anthropic_response: MagicMock,
) -> None:
    """Anthropic should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_anthropic_response,
    )

    anthropic_provider._client.messages.create = (
        create
    )

    await anthropic_provider.generate(
        sample_messages,
        model="claude-test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == "claude-test-model"

    assert request["max_tokens"] == 512

    assert request["temperature"] == 0.2

    assert request["system"] == (
        "You are a helpful assistant."
    )

    assert request["messages"] == [
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_response(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Anthropic should reject an empty text response."""

    response = MagicMock()
    response.content = []

    anthropic_provider._client.messages.create = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no text output",
    ):
        await anthropic_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_ignores_non_text_blocks(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
    mock_anthropic_response: MagicMock,
) -> None:
    """Anthropic should only collect text content blocks."""

    image_block = MagicMock()
    image_block.type = "image"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello"

    mock_anthropic_response.content = [
        image_block,
        text_block,
    ]

    anthropic_provider._client.messages.create = (
        AsyncMock(
            return_value=mock_anthropic_response,
        )
    )

    response = await anthropic_provider.generate(
        sample_messages,
    )

    assert response.content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Anthropic should stream response text."""

    async def fake_text_stream():
        yield "Hello"
        yield " world"

    final_message = MagicMock()
    final_message.model = "claude-test-model"
    final_message.stop_reason = "end_turn"
    final_message.id = "test-response-id"

    stream = MagicMock()

    stream.text_stream = fake_text_stream()
    stream.request_id = "test-request-id"

    stream.get_final_message = AsyncMock(
        return_value=final_message,
    )

    context_manager = MagicMock()

    context_manager.__aenter__ = AsyncMock(
        return_value=stream,
    )
    context_manager.__aexit__ = AsyncMock(
        return_value=None,
    )

    anthropic_provider._client.messages.stream = (
        MagicMock(
            return_value=context_manager,
        )
    )

    chunks = [
        chunk
        async for chunk in anthropic_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 3

    assert chunks[0].content == "Hello"

    assert chunks[0].provider == "anthropic"

    assert chunks[1].content == " world"

    assert chunks[2].content == ""

    assert chunks[2].finish_reason == (
        "end_turn"
    )

    assert chunks[2].metadata == {
        "response_id": "test-response-id",
    }

    stream.get_final_message.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_empty_text(
    anthropic_provider: AnthropicProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Anthropic should skip empty stream text."""

    stream = MagicMock()

    async def fake_text_stream():
        yield ""
        yield "Hello"
        yield ""

    stream.text_stream = fake_text_stream()
    stream.request_id = "test-request-id"

    final_message = MagicMock()
    final_message.model = "claude-test-model"
    final_message.stop_reason = "end_turn"
    final_message.id = "test-response-id"

    stream.get_final_message = AsyncMock(
        return_value=final_message,
    )

    context_manager = MagicMock()

    context_manager.__aenter__ = AsyncMock(
        return_value=stream,
    )
    context_manager.__aexit__ = AsyncMock(
        return_value=None,
    )

    anthropic_provider._client.messages.stream = (
        MagicMock(
            return_value=context_manager,
        )
    )

    chunks = [
        chunk
        async for chunk in anthropic_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 2

    assert chunks[0].content == "Hello"

    assert chunks[1].content == ""


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_not_supported(
    anthropic_provider: AnthropicProvider,
) -> None:
    """Anthropic should reject embedding requests."""

    with pytest.raises(
        ProviderRequestError,
        match="does not provide an embeddings API",
    ):
        await anthropic_provider.embed(
            ["Hello"],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    anthropic_provider: AnthropicProvider,
) -> None:
    """Anthropic health check should return True."""

    anthropic_provider._client.models.retrieve = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await anthropic_provider.health_check()

    assert result is True

    anthropic_provider._client.models.retrieve.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    anthropic_provider: AnthropicProvider,
) -> None:
    """Anthropic health check should return False."""

    anthropic_provider._client.models.retrieve = (
        AsyncMock(
            side_effect=Exception(
                "Anthropic unavailable",
            ),
        )
    )

    result = await anthropic_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    anthropic_provider: AnthropicProvider,
) -> None:
    """Anthropic client should close cleanly."""

    anthropic_provider._client.close = AsyncMock()

    await anthropic_provider.close()

    anthropic_provider._client.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """Anthropic authentication errors should be normalized."""

    error = anthropic.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """Anthropic rate limits should be normalized."""

    error = anthropic.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """Anthropic timeouts should be normalized."""

    error = anthropic.APITimeoutError(
        request=MagicMock(),
    )

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """Anthropic connection errors should be normalized."""

    error = anthropic.APIConnectionError(
        request=MagicMock(),
    )

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_server_error() -> None:
    """Anthropic 5xx errors should be unavailable errors."""

    error = MagicMock(
        spec=anthropic.APIStatusError,
    )

    error.status_code = 500

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_client_error() -> None:
    """Anthropic 4xx errors should be request errors."""

    error = MagicMock(
        spec=anthropic.APIStatusError,
    )

    error.status_code = 400

    result = AnthropicProvider._map_error(
        error,
    )

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "anthropic"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = AnthropicProvider._map_error(
        RuntimeError("Unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "anthropic"