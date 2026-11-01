"""Tests for the Gemini provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
from app.providers.gemini import GeminiProvider


@pytest.fixture
def gemini_provider(
    mock_gemini_client: MagicMock,
) -> GeminiProvider:
    """Return a Gemini provider with a mocked client."""

    with (
        patch(
            "app.providers.gemini.provider.genai.Client",
            return_value=mock_gemini_client,
        ),
        patch(
            "app.providers.gemini.provider.settings.ai.gemini.api_key",
            "test-api-key",
        ),
    ):
        provider = GeminiProvider()

    return provider


@pytest.fixture
def mock_gemini_response() -> MagicMock:
    """Return a mocked Gemini generation response."""

    response = MagicMock()

    response.text = (
        "Hello! I am working correctly."
    )

    response.model_version = "gemini-test-model"

    response.usage_metadata = MagicMock(
        prompt_token_count=20,
        candidates_token_count=10,
        total_token_count=30,
    )

    return response


@pytest.fixture
def mock_gemini_embedding_response() -> MagicMock:
    """Return a mocked Gemini embedding response."""

    first = MagicMock()
    first.values = [
        0.1,
        0.2,
        0.3,
    ]

    second = MagicMock()
    second.values = [
        0.4,
        0.5,
        0.6,
    ]

    response = MagicMock()
    response.embeddings = [
        first,
        second,
    ]

    return response


@pytest.mark.unit
@pytest.mark.provider
def test_provider_name() -> None:
    """Gemini provider should expose the correct name."""

    provider = GeminiProvider.__new__(
        GeminiProvider,
    )

    assert provider.name == "gemini"


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages() -> None:
    """Gemini should convert common messages correctly."""

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

    contents, system_instruction = (
        GeminiProvider._prepare_messages(
            messages,
        )
    )

    assert system_instruction == "You are helpful."

    assert contents == [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hello!",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hi!",
                },
            ],
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_preserves_model_role() -> None:
    """Gemini should preserve the model role."""

    messages = [
        {
            "role": "model",
            "content": "Previous answer.",
        },
    ]

    contents, system_instruction = (
        GeminiProvider._prepare_messages(
            messages,
        )
    )

    assert system_instruction is None

    assert contents == [
        {
            "role": "model",
            "parts": [
                {
                    "text": "Previous answer.",
                },
            ],
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_converts_unknown_role() -> None:
    """Unknown message roles should become user messages."""

    messages = [
        {
            "role": "assistant",
            "content": "Some content.",
        },
    ]

    contents, _ = GeminiProvider._prepare_messages(
        messages,
    )

    assert contents[0]["role"] == "user"


@pytest.mark.unit
@pytest.mark.provider
def test_prepare_messages_supports_content_parts() -> None:
    """Gemini should preserve structured content parts."""

    parts = [
        {
            "text": "First part",
        },
        {
            "text": "Second part",
        },
    ]

    contents, _ = GeminiProvider._prepare_messages(
        [
            {
                "role": "user",
                "content": parts,
            },
        ],
    )

    assert contents == [
        {
            "role": "user",
            "parts": parts,
        },
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_build_config() -> None:
    """Gemini should build generation configuration."""

    with patch(
        "app.providers.gemini.provider.types.GenerateContentConfig",
    ) as config_class:
        provider = GeminiProvider.__new__(
            GeminiProvider,
        )

        result = provider._build_config(
            temperature=0.2,
            max_tokens=512,
            system_instruction="Be concise.",
            kwargs={
                "top_p": 0.9,
            },
        )

    config_class.assert_called_once_with(
        temperature=0.2,
        max_output_tokens=512,
        system_instruction="Be concise.",
        top_p=0.9,
    )

    assert result is config_class.return_value


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    gemini_provider: GeminiProvider,
    sample_messages: list[dict[str, str]],
    mock_gemini_response: MagicMock,
) -> None:
    """Gemini should generate a complete response."""

    create = AsyncMock(
        return_value=mock_gemini_response,
    )

    gemini_provider._client.aio.models.generate_content = (
        create
    )

    response = await gemini_provider.generate(
        sample_messages,
    )

    assert response.content == (
        "Hello! I am working correctly."
    )

    assert response.provider == "gemini"

    assert response.model == (
        "gemini-test-model"
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
    gemini_provider: GeminiProvider,
    sample_messages: list[dict[str, str]],
    mock_gemini_response: MagicMock,
) -> None:
    """Gemini should forward generation parameters."""

    create = AsyncMock(
        return_value=mock_gemini_response,
    )

    gemini_provider._client.aio.models.generate_content = (
        create
    )

    with patch(
        "app.providers.gemini.provider.types.GenerateContentConfig",
    ) as config_class:
        await gemini_provider.generate(
            sample_messages,
            model="gemini-test-model",
            temperature=0.2,
            max_tokens=512,
        )

    request = create.await_args.kwargs

    assert request["model"] == "gemini-test-model"

    assert request["contents"] == [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hello, how are you?",
                },
            ],
        },
    ]

    config_class.assert_called_once()

    config_kwargs = (
        config_class.call_args.kwargs
    )

    assert config_kwargs["temperature"] == 0.2

    assert config_kwargs["max_output_tokens"] == 512

    assert config_kwargs["system_instruction"] == (
        "You are a helpful assistant."
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_empty_response(
    gemini_provider: GeminiProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Gemini should reject an empty response."""

    response = MagicMock()
    response.text = ""

    gemini_provider._client.aio.models.generate_content = (
        AsyncMock(
            return_value=response,
        )
    )

    with pytest.raises(
        ProviderResponseError,
        match="no text output",
    ):
        await gemini_provider.generate(
            sample_messages,
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    gemini_provider: GeminiProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Gemini should stream text chunks."""

    first = MagicMock()
    first.text = "Hello"
    first.model_version = "gemini-test-model"

    second = MagicMock()
    second.text = " world"
    second.model_version = "gemini-test-model"

    async def fake_stream():
        yield first
        yield second

    gemini_provider._client.aio.models.generate_content_stream = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in gemini_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 2

    assert chunks[0].content == "Hello"

    assert chunks[1].content == " world"

    assert chunks[0].provider == "gemini"

    assert chunks[0].model == (
        "gemini-test-model"
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_skips_empty_chunks(
    gemini_provider: GeminiProvider,
    sample_messages: list[dict[str, str]],
) -> None:
    """Gemini should skip chunks without text."""

    empty = MagicMock()
    empty.text = None

    valid = MagicMock()
    valid.text = "Hello"
    valid.model_version = "gemini-test-model"

    async def fake_stream():
        yield empty
        yield valid

    gemini_provider._client.aio.models.generate_content_stream = (
        AsyncMock(
            return_value=fake_stream(),
        )
    )

    chunks = [
        chunk
        async for chunk in gemini_provider.stream(
            sample_messages,
        )
    ]

    assert len(chunks) == 1

    assert chunks[0].content == "Hello"


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed(
    gemini_provider: GeminiProvider,
    sample_embedding_texts: list[str],
    mock_gemini_embedding_response: MagicMock,
) -> None:
    """Gemini should generate embeddings."""

    create = AsyncMock(
        return_value=mock_gemini_embedding_response,
    )

    gemini_provider._client.aio.models.embed_content = (
        create
    )

    with patch(
        "app.providers.gemini.provider.types.EmbedContentConfig",
    ) as config_class:
        result = await gemini_provider.embed(
            sample_embedding_texts,
        )

    assert result.provider == "gemini"

    assert result.model == "gemini-embedding-2"

    assert result.embeddings == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    create.assert_awaited_once()

    request = create.await_args.kwargs

    assert request["model"] == "gemini-embedding-2"

    assert request["contents"] == (
        sample_embedding_texts
    )

    config_class.assert_called_once()

    config_kwargs = (
        config_class.call_args.kwargs
    )

    assert (
        config_kwargs["output_dimensionality"]
        == 1536
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_empty_input(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini should return empty embeddings for empty input."""

    result = await gemini_provider.embed([])

    assert result.provider == "gemini"

    assert result.model == "gemini-embedding-2"

    assert result.embeddings == []


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_rejects_empty_text(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini should reject empty embedding text."""

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        await gemini_provider.embed(
            [
                "hello",
                "",
            ],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_rejects_whitespace_text(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini should reject whitespace-only embedding text."""

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        await gemini_provider.embed(
            [
                "hello",
                "   ",
            ],
        )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini health check should return True."""

    gemini_provider._client.aio.models.get = (
        AsyncMock(
            return_value=MagicMock(),
        )
    )

    result = await gemini_provider.health_check()

    assert result is True

    gemini_provider._client.aio.models.get.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_failure(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini health check should return False on failure."""

    gemini_provider._client.aio.models.get = (
        AsyncMock(
            side_effect=Exception(
                "Gemini unavailable",
            ),
        )
    )

    result = await gemini_provider.health_check()

    assert result is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    gemini_provider: GeminiProvider,
) -> None:
    """Gemini client should close cleanly."""

    gemini_provider._client.aio.aclose = (
        AsyncMock()
    )

    await gemini_provider.close()

    gemini_provider._client.aio.aclose.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
def test_map_authentication_error() -> None:
    """Gemini authentication errors should be normalized."""

    class AuthenticationError(Exception):
        """Fake Gemini authentication error."""

    result = GeminiProvider._map_error(
        AuthenticationError("invalid key"),
    )

    assert isinstance(
        result,
        ProviderAuthenticationError,
    )

    assert result.provider == "gemini"


@pytest.mark.unit
@pytest.mark.provider
def test_map_rate_limit_error() -> None:
    """Gemini rate-limit errors should be normalized."""

    class ResourceExhaustedError(Exception):
        """Fake Gemini quota error."""

    result = GeminiProvider._map_error(
        ResourceExhaustedError("quota exceeded"),
    )

    assert isinstance(
        result,
        ProviderRateLimitError,
    )

    assert result.provider == "gemini"

    assert result.retryable is True


@pytest.mark.unit
@pytest.mark.provider
def test_map_timeout_error() -> None:
    """Gemini timeout errors should be normalized."""

    class TimeoutError(Exception):
        """Fake Gemini timeout error."""

    result = GeminiProvider._map_error(
        TimeoutError("request timed out"),
    )

    assert isinstance(
        result,
        ProviderTimeoutError,
    )

    assert result.retryable is True


@pytest.mark.unit
@pytest.mark.provider
def test_map_connection_error() -> None:
    """Gemini connection errors should be normalized."""

    class ConnectionError(Exception):
        """Fake Gemini connection error."""

    result = GeminiProvider._map_error(
        ConnectionError("connection failed"),
    )

    assert isinstance(
        result,
        ProviderUnavailableError,
    )

    assert result.retryable is True


@pytest.mark.unit
@pytest.mark.provider
def test_map_unavailable_error() -> None:
    """Gemini unavailable errors should be normalized."""

    class UnavailableError(Exception):
        """Fake Gemini unavailable error."""

    result = GeminiProvider._map_error(
        UnavailableError("service unavailable"),
    )

    assert isinstance(
        result,
        ProviderUnavailableError,
    )


@pytest.mark.unit
@pytest.mark.provider
def test_map_invalid_argument_error() -> None:
    """Gemini invalid requests should be normalized."""

    class InvalidArgumentError(Exception):
        """Fake Gemini invalid argument error."""

    result = GeminiProvider._map_error(
        InvalidArgumentError("invalid request"),
    )

    assert isinstance(
        result,
        ProviderRequestError,
    )

    assert result.provider == "gemini"


@pytest.mark.unit
@pytest.mark.provider
def test_map_unknown_error() -> None:
    """Unknown errors should become ProviderError."""

    result = GeminiProvider._map_error(
        RuntimeError("unexpected failure"),
    )

    assert isinstance(
        result,
        ProviderError,
    )

    assert result.provider == "gemini"