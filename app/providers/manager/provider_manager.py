"""Provider manager."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.settings import settings
from app.providers.base.exceptions import (
    ProviderConfigurationError,
    ProviderError,
)
from app.providers.base.provider import BaseProvider
from app.providers.manager.demo_fallback import DemoFallbackProvider
from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
)


class ProviderManager:
    """Manage AI providers and provider fallback."""

    def __init__(
        self,
        providers: dict[str, BaseProvider] | None = None,
    ) -> None:
        self._providers: dict[str, BaseProvider] = (
            providers or {}
        )

        if not self._providers:
            self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register configured providers."""

        provider_targets = {
            "openai": ("app.providers.openai", "OpenAIProvider"),
            "anthropic": ("app.providers.anthropic", "AnthropicProvider"),
            "gemini": ("app.providers.gemini", "GeminiProvider"),
            "groq": ("app.providers.groq", "GroqProvider"),
            "deepseek": ("app.providers.deepseek", "DeepSeekProvider"),
            "qwen": ("app.providers.qwen", "QwenProvider"),
            "openrouter": ("app.providers.openrouter", "OpenRouterProvider"),
        }

        for name, (module_name, class_name) in provider_targets.items():
            try:
                module = __import__(module_name, fromlist=[class_name])
                provider_class = getattr(module, class_name)
                provider = provider_class()
            except (ImportError, AttributeError, ProviderError):
                # Providers are optional. A missing SDK or missing key must
                # not prevent the local development server from starting.
                continue

            self._providers[name] = provider

        # Keep the local development UI usable even before external provider
        # credentials are configured. This fallback is never registered in
        # production.
        if settings.is_development:
            self._providers.setdefault("demo_fallback", DemoFallbackProvider())

    @property
    def providers(self) -> dict[str, BaseProvider]:
        """Return registered providers."""

        return self._providers.copy()

    def register(
        self,
        provider: BaseProvider,
    ) -> None:
        """Register or replace a provider."""

        self._providers[provider.name] = provider

    def unregister(
        self,
        name: str,
    ) -> BaseProvider | None:
        """Remove and return a registered provider."""

        return self._providers.pop(name, None)

    def get(
        self,
        name: str,
    ) -> BaseProvider:
        """Return a provider by name."""

        try:
            return self._providers[name]
        except KeyError as error:
            raise ProviderConfigurationError(
                f"Provider '{name}' is not registered.",
                provider=name,
            ) from error

    def _fallback_order(
        self,
        primary: str,
    ) -> list[str]:
        """Build the configured provider fallback order."""

        fallback = settings.ai.fallback_provider

        order = [primary]

        if fallback and fallback != primary:
            order.append(fallback)

        return [
            name
            for name in order
            if name in self._providers
        ]

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        fallback: bool = True,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response using the selected provider."""

        primary = (
            provider
            or settings.ai.default_provider
        )

        # In local development, a requested provider may be unavailable
        # because its SDK/key is not configured. Use the clearly-labelled
        # local demo fallback instead of making the demo unusable.
        if (
            provider is not None
            and provider not in self._providers
        ):
            if settings.is_development and "demo_fallback" in self._providers:
                fallback_provider = self._providers["demo_fallback"]
                return await fallback_provider.generate(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            raise ProviderConfigurationError(
                f"Provider '{provider}' is not registered.",
                provider=provider,
            )

        providers = (
            self._fallback_order(primary)
            if fallback
            else [primary]
        )

        errors: list[ProviderError] = []

        for name in providers:
            current = self.get(name)

            try:
                return await current.generate(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

            except ProviderError as error:
                errors.append(error)

                if not error.retryable:
                    raise

        if settings.is_development and "demo_fallback" in self._providers:
            fallback_provider = self._providers["demo_fallback"]
            return await fallback_provider.generate(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        raise ProviderError(
            "All configured providers failed.",
            provider=primary,
            retryable=False,
        ) from (
            errors[-1]
            if errors
            else None
        )

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderChunk]:
        """Stream a response from the selected provider."""

        primary = (
            provider
            or settings.ai.default_provider
        )

        current = self.get(primary)

        async for chunk in current.stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    async def embed(
        self,
        texts: Sequence[str],
        *,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings using the configured provider."""

        primary = (
            provider
            or settings.ai.embedding_provider
        )

        current = self.get(primary)

        return await current.embed(
            texts,
            model=model,
            **kwargs,
        )

    async def health_check(
        self,
        provider: str | None = None,
    ) -> dict[str, bool]:
        """Check the health of registered providers."""

        if provider:
            current = self.get(provider)

            return {
                provider: await current.health_check(),
            }

        results: dict[str, bool] = {}

        for name, current in self._providers.items():
            try:
                results[name] = await current.health_check()
            except Exception:
                results[name] = False

        return results

    async def close(self) -> None:
        """Close all registered providers."""

        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception:
                continue