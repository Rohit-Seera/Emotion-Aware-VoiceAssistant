"""Tests for the provider manager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.providers.base.exceptions import (
    ProviderConfigurationError,
    ProviderError,
)
from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
)
from app.providers.manager.provider_manager import ProviderManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def primary_provider() -> MagicMock:
    """Return a mocked primary provider."""

    provider = MagicMock()
    provider.name = "openai"

    provider.generate = AsyncMock(
        return_value=ProviderResponse(
            content="Primary response",
            provider="openai",
            model="test-model",
        )
    )

    provider.stream = MagicMock()
    provider.embed = AsyncMock(
        return_value=EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            provider="openai",
            model="test-embedding-model",
        )
    )

    provider.health_check = AsyncMock(
        return_value=True,
    )
    provider.close = AsyncMock()

    return provider


@pytest.fixture
def fallback_provider() -> MagicMock:
    """Return a mocked fallback provider."""

    provider = MagicMock()
    provider.name = "gemini"

    provider.generate = AsyncMock(
        return_value=ProviderResponse(
            content="Fallback response",
            provider="gemini",
            model="fallback-model",
        )
    )

    provider.stream = MagicMock()
    provider.embed = AsyncMock()

    provider.health_check = AsyncMock(
        return_value=True,
    )
    provider.close = AsyncMock()

    return provider


@pytest.fixture
def manager(
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
) -> ProviderManager:
    """Return a manager with mocked providers."""

    return ProviderManager(
        providers={
            "openai": primary_provider,
            "gemini": fallback_provider,
        }
    )


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Return sample messages."""

    return [
        {
            "role": "user",
            "content": "Hello!",
        }
    ]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_providers_returns_copy(
    manager: ProviderManager,
) -> None:
    """Providers property should return a copy."""

    providers = manager.providers

    assert providers is not manager._providers
    assert providers == manager._providers


@pytest.mark.unit
@pytest.mark.provider
def test_register(
    manager: ProviderManager,
) -> None:
    """Manager should register a provider."""

    provider = MagicMock()
    provider.name = "test"

    manager.register(provider)

    assert manager.get("test") is provider


@pytest.mark.unit
@pytest.mark.provider
def test_register_replaces_existing_provider(
    manager: ProviderManager,
) -> None:
    """Register should replace an existing provider."""

    provider = MagicMock()
    provider.name = "openai"

    manager.register(provider)

    assert manager.get("openai") is provider


@pytest.mark.unit
@pytest.mark.provider
def test_unregister(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Manager should remove and return a provider."""

    result = manager.unregister("openai")

    assert result is primary_provider
    assert "openai" not in manager.providers


@pytest.mark.unit
@pytest.mark.provider
def test_unregister_missing_provider(
    manager: ProviderManager,
) -> None:
    """Unregistering a missing provider should return None."""

    result = manager.unregister("missing")

    assert result is None


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_get(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Manager should return a registered provider."""

    result = manager.get("openai")

    assert result is primary_provider


@pytest.mark.unit
@pytest.mark.provider
def test_get_missing_provider(
    manager: ProviderManager,
) -> None:
    """Getting an unknown provider should raise configuration error."""

    with pytest.raises(
        ProviderConfigurationError,
        match="Provider 'missing' is not registered",
    ):
        manager.get("missing")


# ---------------------------------------------------------------------------
# Fallback order
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_fallback_order(
    manager: ProviderManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback order should contain primary and configured fallback."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.fallback_provider",
        "gemini",
    )

    result = manager._fallback_order("openai")

    assert result == [
        "openai",
        "gemini",
    ]


@pytest.mark.unit
@pytest.mark.provider
def test_fallback_order_does_not_duplicate_primary(
    manager: ProviderManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback should not duplicate the primary provider."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.fallback_provider",
        "openai",
    )

    result = manager._fallback_order("openai")

    assert result == ["openai"]


@pytest.mark.unit
@pytest.mark.provider
def test_fallback_order_ignores_unregistered_fallback(
    manager: ProviderManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unregistered fallback providers should be ignored."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.fallback_provider",
        "anthropic",
    )

    result = manager._fallback_order("openai")

    assert result == ["openai"]


@pytest.mark.unit
@pytest.mark.provider
def test_fallback_order_without_fallback(
    manager: ProviderManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No configured fallback should return only the primary."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.fallback_provider",
        None,
    )

    result = manager._fallback_order("openai")

    assert result == ["openai"]


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate(
    manager: ProviderManager,
    primary_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should generate using the primary provider."""

    result = await manager.generate(
        sample_messages,
        provider="openai",
    )

    assert result.content == "Primary response"
    assert result.provider == "openai"

    primary_provider.generate.assert_awaited_once_with(
        sample_messages,
        model=None,
        temperature=None,
        max_tokens=None,
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_uses_default_provider(
    manager: ProviderManager,
    primary_provider: MagicMock,
    sample_messages: list[dict[str, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manager should use the configured default provider."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.default_provider",
        "openai",
    )

    result = await manager.generate(
        sample_messages,
    )

    assert result.provider == "openai"

    primary_provider.generate.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_passes_parameters(
    manager: ProviderManager,
    primary_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should forward generation parameters."""

    await manager.generate(
        sample_messages,
        provider="openai",
        model="test-model",
        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
    )

    primary_provider.generate.assert_awaited_once_with(
        sample_messages,
        model="test-model",
        temperature=0.2,
        max_tokens=512,
        top_p=0.9,
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_fallback_on_retryable_error(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should use fallback after a retryable error."""

    primary_provider.generate = AsyncMock(
        side_effect=ProviderError(
            "Temporary failure",
            provider="openai",
            retryable=True,
        )
    )

    result = await manager.generate(
        sample_messages,
        provider="openai",
    )

    assert result.content == "Fallback response"
    assert result.provider == "gemini"

    primary_provider.generate.assert_awaited_once()
    fallback_provider.generate.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_does_not_fallback_on_non_retryable_error(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should not fallback on a non-retryable error."""

    error = ProviderError(
        "Permanent failure",
        provider="openai",
        retryable=False,
    )

    primary_provider.generate = AsyncMock(
        side_effect=error,
    )

    with pytest.raises(ProviderError) as exc_info:
        await manager.generate(
            sample_messages,
            provider="openai",
        )

    assert exc_info.value is error

    primary_provider.generate.assert_awaited_once()
    fallback_provider.generate.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_without_fallback(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Fallback=False should disable fallback."""

    primary_provider.generate = AsyncMock(
        side_effect=ProviderError(
            "Temporary failure",
            provider="openai",
            retryable=True,
        )
    )

    with pytest.raises(
        ProviderError,
        match="All configured providers failed",
    ):
        await manager.generate(
            sample_messages,
            provider="openai",
            fallback=False,
        )

    fallback_provider.generate.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_all_providers_fail(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should raise when all providers fail."""

    primary_provider.generate = AsyncMock(
        side_effect=ProviderError(
            "Primary failed",
            provider="openai",
            retryable=True,
        )
    )

    fallback_provider.generate = AsyncMock(
        side_effect=ProviderError(
            "Fallback failed",
            provider="gemini",
            retryable=True,
        )
    )

    with pytest.raises(
        ProviderError,
        match="All configured providers failed",
    ) as exc_info:
        await manager.generate(
            sample_messages,
            provider="openai",
        )

    assert exc_info.value.provider == "openai"

    assert primary_provider.generate.await_count == 1
    assert fallback_provider.generate.await_count == 1


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_generate_missing_primary_provider(
    manager: ProviderManager,
    sample_messages: list[dict[str, str]],
) -> None:
    """Generate should fail when selected provider is missing."""

    with pytest.raises(
        ProviderConfigurationError,
        match="Provider 'missing' is not registered",
    ):
        await manager.generate(
            sample_messages,
            provider="missing",
        )


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream(
    manager: ProviderManager,
    primary_provider: MagicMock,
    sample_messages: list[dict[str, str]],
) -> None:
    """Manager should stream from the selected provider."""

    chunk = ProviderChunk(
        content="Hello",
        provider="openai",
        model="test-model",
    )

    async def fake_stream() -> AsyncIterator[ProviderChunk]:
        yield chunk

    primary_provider.stream = MagicMock(
        return_value=fake_stream(),
    )

    chunks = [
        item
        async for item in manager.stream(
            sample_messages,
            provider="openai",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"
    assert chunks[0].provider == "openai"

    primary_provider.stream.assert_called_once_with(
        sample_messages,
        model=None,
        temperature=None,
        max_tokens=None,
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_stream_missing_provider(
    manager: ProviderManager,
    sample_messages: list[dict[str, str]],
) -> None:
    """Stream should fail for an unknown provider."""

    with pytest.raises(
        ProviderConfigurationError,
        match="Provider 'missing' is not registered",
    ):
        async for _ in manager.stream(
            sample_messages,
            provider="missing",
        ):
            pass


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Manager should generate embeddings."""

    texts = ["Hello", "World"]

    result = await manager.embed(
        texts,
        provider="openai",
        model="test-embedding-model",
    )

    assert result.provider == "openai"
    assert result.embeddings == [
        [0.1, 0.2, 0.3],
    ]

    primary_provider.embed.assert_awaited_once_with(
        texts,
        model="test-embedding-model",
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_uses_default_provider(
    manager: ProviderManager,
    primary_provider: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embed should use the configured embedding provider."""

    monkeypatch.setattr(
        "app.providers.manager.provider_manager.settings.ai.embedding_provider",
        "openai",
    )

    result = await manager.embed(
        ["Hello"],
    )

    assert result.provider == "openai"

    primary_provider.embed.assert_awaited_once_with(
        ["Hello"],
        model=None,
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_passes_extra_kwargs(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Manager should forward embedding kwargs."""

    await manager.embed(
        ["Hello"],
        provider="openai",
        model="test-model",
        dimensions=256,
    )

    primary_provider.embed.assert_awaited_once_with(
        ["Hello"],
        model="test-model",
        dimensions=256,
    )


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_embed_missing_provider(
    manager: ProviderManager,
) -> None:
    """Embed should fail for an unknown provider."""

    with pytest.raises(
        ProviderConfigurationError,
        match="Provider 'missing' is not registered",
    ):
        await manager.embed(
            ["Hello"],
            provider="missing",
        )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_single_provider(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Health check should support a specific provider."""

    result = await manager.health_check(
        provider="openai",
    )

    assert result == {
        "openai": True,
    }

    primary_provider.health_check.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_all_providers(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
) -> None:
    """Health check should check all providers."""

    fallback_provider.health_check = AsyncMock(
        return_value=False,
    )

    result = await manager.health_check()

    assert result == {
        "openai": True,
        "gemini": False,
    }

    primary_provider.health_check.assert_awaited_once()
    fallback_provider.health_check.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_handles_exception(
    manager: ProviderManager,
    primary_provider: MagicMock,
) -> None:
    """Health check should convert exceptions to False."""

    primary_provider.health_check = AsyncMock(
        side_effect=RuntimeError(
            "Health check failed",
        ),
    )

    result = await manager.health_check()

    assert result["openai"] is False


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_health_check_missing_provider(
    manager: ProviderManager,
) -> None:
    """Health check should fail for an unknown provider."""

    with pytest.raises(
        ProviderConfigurationError,
        match="Provider 'missing' is not registered",
    ):
        await manager.health_check(
            provider="missing",
        )


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
) -> None:
    """Manager should close all providers."""

    await manager.close()

    primary_provider.close.assert_awaited_once()
    fallback_provider.close.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.provider
@pytest.mark.asyncio
async def test_close_continues_after_provider_failure(
    manager: ProviderManager,
    primary_provider: MagicMock,
    fallback_provider: MagicMock,
) -> None:
    """Close should continue when one provider fails."""

    primary_provider.close = AsyncMock(
        side_effect=RuntimeError(
            "Close failed",
        ),
    )

    await manager.close()

    primary_provider.close.assert_awaited_once()
    fallback_provider.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Default provider registration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.provider
def test_register_default_providers_skips_configuration_errors() -> None:
    """Default registration should skip unavailable providers."""

    successful_provider = MagicMock()
    successful_provider.name = "openai"

    with patch(
        "app.providers.manager.provider_manager.OpenAIProvider",
        return_value=successful_provider,
    ), patch(
        "app.providers.manager.provider_manager.AnthropicProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="anthropic",
        ),
    ), patch(
        "app.providers.manager.provider_manager.GeminiProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="gemini",
        ),
    ), patch(
        "app.providers.manager.provider_manager.GroqProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="groq",
        ),
    ), patch(
        "app.providers.manager.provider_manager.DeepSeekProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="deepseek",
        ),
    ), patch(
        "app.providers.manager.provider_manager.QwenProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="qwen",
        ),
    ), patch(
        "app.providers.manager.provider_manager.OpenRouterProvider",
        side_effect=ProviderConfigurationError(
            "Missing API key",
            provider="openrouter",
        ),
    ):
        manager = ProviderManager()

    assert manager.providers == {
        "openai": successful_provider,
    }