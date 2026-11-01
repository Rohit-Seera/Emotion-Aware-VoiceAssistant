"""OpenRouter provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import openai
from openai import AsyncOpenAI

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
from app.providers.base.provider import BaseProvider
from app.providers.base.types import (
    EmbeddingResult,
    ProviderChunk,
    ProviderResponse,
    ProviderUsage,
)


class OpenRouterProvider(BaseProvider):
    """OpenRouter implementation of the common provider interface."""

    name = "openrouter"

    def __init__(self) -> None:
        config = settings.ai.openrouter

        if not config.enabled:
            raise ProviderConfigurationError(
                "OpenRouter provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "OpenRouter API key is not configured.",
                provider=self.name,
            )

        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Emotion Aware Voice Assistant",
            },
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert OpenRouter usage metadata."""

        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            0,
        )

        return ProviderUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _map_error(error: Exception) -> ProviderError:
        """Convert OpenAI-compatible errors to common provider errors."""

        if isinstance(error, openai.AuthenticationError):
            return ProviderAuthenticationError(
                "OpenRouter authentication failed.",
                provider="openrouter",
            )

        if isinstance(error, openai.RateLimitError):
            return ProviderRateLimitError(
                "OpenRouter rate limit or quota was exceeded.",
                provider="openrouter",
            )

        if isinstance(error, openai.APITimeoutError):
            return ProviderTimeoutError(
                "OpenRouter request timed out.",
                provider="openrouter",
            )

        if isinstance(error, openai.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to OpenRouter.",
                provider="openrouter",
            )

        if isinstance(error, openai.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"OpenRouter service returned status "
                    f"{status_code}.",
                    provider="openrouter",
                )

            return ProviderRequestError(
                f"OpenRouter request failed with status "
                f"{status_code}.",
                provider="openrouter",
            )

        return ProviderError(
            f"Unexpected OpenRouter provider error: {error}",
            provider="openrouter",
        )

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a complete response through OpenRouter."""

        config = settings.ai.openrouter

        if not model:
            raise ProviderConfigurationError(
                "OpenRouter requires an explicit model.",
                provider=self.name,
            )

        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": (
                temperature
                if temperature is not None
                else config.temperature
            ),
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
        }

        request.update(kwargs)

        try:
            response = await self._client.chat.completions.create(
                **request
            )
        except Exception as error:
            raise self._map_error(error) from error

        if not response.choices:
            raise ProviderResponseError(
                "OpenRouter returned no choices.",
                provider=self.name,
            )

        choice = response.choices[0]

        return ProviderResponse(
            content=choice.message.content or "",
            provider=self.name,
            model=response.model,
            usage=self._usage(response),
            finish_reason=choice.finish_reason,
            request_id=getattr(
                response,
                "_request_id",
                None,
            ),
            metadata={
                "response_id": getattr(
                    response,
                    "id",
                    None,
                ),
            },
        )

    async def stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ProviderChunk]:
        """Stream a response through OpenRouter."""

        config = settings.ai.openrouter

        if not model:
            raise ProviderConfigurationError(
                "OpenRouter requires an explicit model.",
                provider=self.name,
            )

        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": (
                temperature
                if temperature is not None
                else config.temperature
            ),
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
            "stream": True,
        }

        request.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(
                **request
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content or ""

                if not content:
                    continue

                yield ProviderChunk(
                    content=content,
                    provider=self.name,
                    model=(
                        chunk.model
                        or model
                    ),
                    request_id=getattr(
                        chunk,
                        "_request_id",
                        None,
                    ),
                    finish_reason=choice.finish_reason,
                )

        except Exception as error:
            raise self._map_error(error) from error

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings through OpenRouter."""

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                provider=self.name,
                model=model or "",
            )

        if not model:
            raise ProviderConfigurationError(
                "OpenRouter embedding requires an explicit model.",
                provider=self.name,
            )

        request: dict[str, Any] = {
            "model": model,
            "input": list(texts),
        }

        request.update(kwargs)

        try:
            response = await self._client.embeddings.create(
                **request
            )
        except Exception as error:
            raise self._map_error(error) from error

        embeddings = [
            item.embedding
            for item in response.data
        ]

        return EmbeddingResult(
            embeddings=embeddings,
            provider=self.name,
            model=response.model,
            usage=self._usage(response),
        )

    async def health_check(self) -> bool:
        """Check whether OpenRouter is reachable."""

        try:
            await self._client.models.list()
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the OpenRouter client."""

        await self._client.close()
