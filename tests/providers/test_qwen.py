"""Tests for the Qwen provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from app.core.settings import settings
from app.providers.base.exceptions import (
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.qwen import QwenProvider


@pytest.fixture
def qwen_provider(
    mock_async_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> QwenProvider:
    """Return a Qwen provider with isolated test settings."""

    config = settings.ai.qwen

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
        "model",
        "qwen-plus",
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

    monkeypatch.setattr(
        "app.providers.qwen.provider.AsyncOpenAI",
        lambda **kwargs: mock_async_client,
    )

    return QwenProvider()


@pytest.fixture
def mock_qwen_response() -> MagicMock:
    """Return a mocked Qwen chat completion response."""

    message = MagicMock()
    message.content = (
        "Hello! I am working correctly."
    )
    message.role = "assistant"
    message.reasoning_content = (
        "I should respond helpfully."
    )

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.id = "test-response-id"
    response.model = "qwen-plus"
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
    """Qwen provider should expose the correct name."""

    provider = QwenProvider.__new__(
        QwenProvider,
    )

    assert provider.name == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_usage() -> None:
    """Qwen usage should be converted correctly."""

    response = MagicMock()

    response.usage = MagicMock(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )

    result = QwenProvider._usage(response)

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

    result = QwenProvider._usage(response)

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

    result = QwenProvider._usage(response)

    assert result is not None
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
    mock_qwen_response: MagicMock,
) -> None:
    """Qwen should generate a complete response."""

    create = AsyncMock(
        return_value=mock_qwen_response,
    )

    qwen_provider._client.chat.completions.create = (
        create
    )

    response = await qwen_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "qwen"
    assert response.model == "qwen-plus"
    assert response.finish_reason == "stop"
    assert response.request_id == "test-request-id"

    assert response.metadata == {
        "response_id": "test-response-id",
        "reasoning_content": (
            "I should respond helpfully."
        ),
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
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
    mock_qwen_response: MagicMock,
) -> None:
    """Qwen should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_qwen_response,
    )

    qwen_provider._client.chat.completions.create = (
        create
    )

    await qwen_provider.generate(
        sample_messages,
        model="qwen-test-model",
        temperature=0.2,
        max_tokens=512,
    )

    request = create.await_args.kwargs

    assert request["model"] == "qwen-test-model"
    assert request["messages"] == sample_messages
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 512


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_uses_config_defaults(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
    mock_qwen_response: MagicMock,
) -> None:
    """Qwen should use configured defaults."""

    create = AsyncMock(
        return_value=mock_qwen_response,
    )

    qwen_provider._client.chat.completions.create = (
        create
    )

    await qwen_provider.generate(
        sample_messages,
    )

    request = create.await_args.kwargs

    assert request["model"] == "qwen-plus"
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 4096


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_extra_kwargs(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
    mock_qwen_response: MagicMock,
) -> None:
    """Qwen should forward additional arguments."""

    create = AsyncMock(
        return_value=mock_qwen_response,
    )

    qwen_provider._client.chat.completions.create = (
        create
    )

    await qwen_provider.generate(
        sample_messages,
        top_p=0.9,
    )

    request = create.await_args.kwargs

    assert request["top_p"] == 0.9


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_no_choices(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should reject responses without choices."""

    response = MagicMock()
    response.choices = []

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no choices",
    ):
        await qwen_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_content(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should handle empty message content."""

    message = MagicMock()
    message.content = None
    message.reasoning_content = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "qwen-plus"
    response.id = "test-response-id"
    response._request_id = "test-request-id"
    response.usage = None

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    result = await qwen_provider.generate(
        sample_messages,
    )

    assert result.content == ""

    assert result.metadata == {
        "response_id": "test-response-id",
        "reasoning_content": None,
    }


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_reasoning_content(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should preserve reasoning content in metadata."""

    message = MagicMock()
    message.content = "Final answer"
    message.reasoning_content = "Internal reasoning"

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"

    response = MagicMock()
    response.choices = [choice]
    response.model = "qwen-plus"
    response.id = "test-response-id"
    response._request_id = "test-request-id"
    response.usage = None

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=response,
        )
    )

    result = await qwen_provider.generate(
        sample_messages,
    )

    assert result.content == "Final answer"

    assert result.metadata["reasoning_content"] == (
        "Internal reasoning"
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should stream response chunks."""

    delta = MagicMock()
    delta.content = "Hello"
    delta.reasoning_content = None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = "qwen-plus"
    chunk._request_id = "test-request-id"

    async def fake_stream():
        yield chunk

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in qwen_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"
    assert chunks[0].provider == "qwen"

    request = (
        qwen_provider
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
async def test_stream_reasoning_content(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should preserve reasoning content in stream metadata."""

    delta = MagicMock()
    delta.content = ""
    delta.reasoning_content = "Thinking..."

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.model = "qwen-plus"

    async def fake_stream():
        yield chunk

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in qwen_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == ""

    assert chunks[0].metadata == {
        "reasoning_content": "Thinking...",
    }


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_empty_chunks(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should skip chunks without content or reasoning."""

    delta = MagicMock()
    delta.content = ""
    delta.reasoning_content = None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    empty_chunk = MagicMock()
    empty_chunk.choices = [choice]

    valid_delta = MagicMock()
    valid_delta.content = "Hello"
    valid_delta.reasoning_content = None

    valid_choice = MagicMock()
    valid_choice.delta = valid_delta
    valid_choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [valid_choice]
    valid_chunk.model = "qwen-plus"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in qwen_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_chunks_without_choices(
    qwen_provider: QwenProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Qwen should skip chunks without choices."""

    empty_chunk = MagicMock()
    empty_chunk.choices = []

    delta = MagicMock()
    delta.content = "Hello"
    delta.reasoning_content = None

    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = None

    valid_chunk = MagicMock()
    valid_chunk.choices = [choice]
    valid_chunk.model = "qwen-plus"

    async def fake_stream():
        yield empty_chunk
        yield valid_chunk

    qwen_provider._client.chat.completions.create = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in qwen_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_not_supported(
    qwen_provider: QwenProvider,
) -> None:
    """Qwen should reject embedding requests."""

    with pytest.raises(
        ProviderRequestError,
        match="embedding support is not configured",
    ):
        await qwen_provider.embed(
            ["Hello"],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    qwen_provider: QwenProvider,
) -> None:
    """Qwen health check should return True."""

    qwen_provider._client.models.retrieve = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await qwen_provider.health_check()

    assert result is True

    qwen_provider._client.models.retrieve.assert_awaited_once_with(
        "qwen-plus",
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    qwen_provider: QwenProvider,
) -> None:
    """Qwen health check should return False."""

    qwen_provider._client.models.retrieve = (
        AsyncMock(
            side_effect=Exception(
                "Qwen unavailable",
            ),
        )
    )

    result = await qwen_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    qwen_provider: QwenProvider,
) -> None:
    """Qwen client should close cleanly."""

    qwen_provider._client.close = AsyncMock()

    await qwen_provider.close()

    qwen_provider._client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """Qwen authentication errors should be normalized."""

    error = openai.AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body=None,
    )

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """Qwen rate-limit errors should be normalized."""

    error = openai.RateLimitError(
        "Rate limit exceeded",
        response=MagicMock(),
        body=None,
    )

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """Qwen timeout errors should be normalized."""

    error = openai.APITimeoutError(
        request=MagicMock(),
    )

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """Qwen connection errors should be normalized."""

    error = openai.APIConnectionError(
        request=MagicMock(),
    )

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_server_error() -> None:
    """Qwen 5xx errors should be unavailable errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 500

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_client_error() -> None:
    """Qwen 4xx errors should be request errors."""

    error = MagicMock(
        spec=openai.APIStatusError,
    )

    error.status_code = 400

    result = QwenProvider._map_error(error)

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "qwen"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = QwenProvider._map_error(
        RuntimeError("Unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "qwen"