"""Shared fixtures for provider tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
    ProviderUsage,
)


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Return a basic conversation for provider tests."""

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "Hello, how are you?",
        },
    ]


@pytest.fixture
def sample_user_message() -> list[dict[str, str]]:
    """Return a minimal user conversation."""

    return [
        {
            "role": "user",
            "content": "Hello!",
        },
    ]


@pytest.fixture
def sample_embedding_texts() -> list[str]:
    """Return sample texts for embedding tests."""

    return [
        "Emotion Aware Voice Assistant is an AI assistant.",
        "The assistant can remember conversations.",
        "Voice interaction is part of the system.",
    ]


@pytest.fixture
def sample_usage() -> ProviderUsage:
    """Return deterministic token usage."""

    return ProviderUsage(
        prompt_tokens=20,
        completion_tokens=10,
        total_tokens=30,
    )


@pytest.fixture
def sample_response(
    sample_usage: ProviderUsage,
) -> ProviderResponse:
    """Return a standard provider response."""

    return ProviderResponse(
        content="Hello! I am working correctly.",
        provider="test-provider",
        model="test-model",
        usage=sample_usage,
        finish_reason="stop",
        request_id="test-request-id",
    )


@pytest.fixture
def sample_chunk() -> ProviderChunk:
    """Return a standard streaming chunk."""

    return ProviderChunk(
        content="Hello",
        provider="test-provider",
        model="test-model",
        request_id="test-request-id",
    )


@pytest.fixture
def sample_final_chunk() -> ProviderChunk:
    """Return the final streaming chunk."""

    return ProviderChunk(
        content="",
        provider="test-provider",
        model="test-model",
        request_id="test-request-id",
        finish_reason="stop",
    )


@pytest.fixture
def sample_embedding_result(
    sample_usage: ProviderUsage,
) -> EmbeddingResult:
    """Return a standard embedding result."""

    return EmbeddingResult(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        provider="test-provider",
        model="test-embedding-model",
        usage=sample_usage,
    )


# ---------------------------------------------------------------------------
# OpenAI fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_response(
    sample_usage: ProviderUsage,
) -> MagicMock:
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
        input_tokens=sample_usage.prompt_tokens,
        output_tokens=sample_usage.completion_tokens,
        total_tokens=sample_usage.total_tokens,
    )

    return response


@pytest.fixture
def mock_embedding_response(
    sample_usage: ProviderUsage,
) -> MagicMock:
    """Return a mocked OpenAI embedding response."""

    first = MagicMock()
    first.index = 0
    first.embedding = [0.1, 0.2, 0.3]

    second = MagicMock()
    second.index = 1
    second.embedding = [0.4, 0.5, 0.6]

    response = MagicMock()
    response.model = "test-embedding-model"
    response.data = [first, second]
    response.usage = sample_usage

    return response


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Return a mocked OpenAI async client."""

    client = MagicMock()

    # OpenAI Responses API.
    client.responses = MagicMock()
    client.responses.create = AsyncMock()

    # OpenAI Embeddings API.
    client.embeddings = MagicMock()
    client.embeddings.create = AsyncMock()

    # OpenAI Models API.
    client.models = MagicMock()
    client.models.retrieve = AsyncMock()
    client.models.list = AsyncMock()

    # OpenAI client lifecycle.
    client.close = AsyncMock()

    return client


@pytest.fixture
def async_stream() -> AsyncIterator[MagicMock]:
    """Return a deterministic OpenAI Responses API stream."""

    delta_event = MagicMock()
    delta_event.type = (
        "response.output_text.delta"
    )
    delta_event.delta = "Hello"

    done_event = MagicMock()
    done_event.type = (
        "response.output_text.done"
    )

    response = MagicMock()
    response.model = "test-model"
    response.id = "test-response-id"
    response._request_id = "test-request-id"

    completed_event = MagicMock()
    completed_event.type = (
        "response.completed"
    )
    completed_event.response = response

    async def _stream() -> AsyncIterator[MagicMock]:
        yield delta_event
        yield done_event
        yield completed_event

    return _stream()


# ---------------------------------------------------------------------------
# Gemini fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gemini_client() -> MagicMock:
    """Return a mocked Google Gemini async client."""

    client = MagicMock()

    client.aio = MagicMock()
    client.aio.models = MagicMock()

    # Gemini generation.
    client.aio.models.generate_content = AsyncMock()

    # Gemini streaming generation.
    client.aio.models.generate_content_stream = (
        AsyncMock()
    )

    # Gemini embeddings.
    client.aio.models.embed_content = AsyncMock()

    # Gemini health check.
    client.aio.models.get = AsyncMock()

    # Gemini client lifecycle.
    client.aio.aclose = AsyncMock()

    return client


# ---------------------------------------------------------------------------
# Anthropic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Return a mocked Anthropic async client."""

    client = MagicMock()

    # Anthropic Messages API.
    client.messages = MagicMock()
    client.messages.create = AsyncMock()

    # Anthropic streaming API.
    #
    # This MUST be a MagicMock rather than AsyncMock because
    # the provider uses:
    #
    # async with client.messages.stream(...) as stream:
    #
    client.messages.stream = MagicMock()

    # Anthropic Models API.
    client.models = MagicMock()
    client.models.retrieve = AsyncMock()

    # Anthropic client lifecycle.
    client.close = AsyncMock()

    return client


# ---------------------------------------------------------------------------
# Groq fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_groq_client() -> MagicMock:
    """Return a mocked Groq async client."""

    client = MagicMock()

    # Groq Chat Completions API.
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()

    # Groq Models API.
    client.models = MagicMock()
    client.models.retrieve = AsyncMock()

    # Groq client lifecycle.
    client.close = AsyncMock()

    return client


# ---------------------------------------------------------------------------
# Common provider configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_provider_config() -> dict[str, Any]:
    """Return common provider configuration values."""

    return {
        "enabled": True,
        "api_key": "test-api-key",
        "base_url": None,
        "model": "test-model",
        "timeout": 30,
        "max_retries": 2,
        "temperature": 0.7,
        "max_tokens": 1024,
    }