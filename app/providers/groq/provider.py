"""Groq provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import groq
from groq import AsyncGroq

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


class GroqProvider(BaseProvider):
    """Groq implementation of the common provider interface."""

    name = "groq"

    def __init__(self) -> None:
        config = settings.ai.groq

        if not config.enabled:
            raise ProviderConfigurationError(
                "Groq provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "Groq API key is not configured.",
                provider=self.name,
            )

        self._client = AsyncGroq(
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert Groq usage metadata."""

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
        """Convert Groq SDK errors to common provider errors."""

        if isinstance(error, groq.AuthenticationError):
            return ProviderAuthenticationError(
                "Groq authentication failed.",
                provider="groq",
            )

        if isinstance(error, groq.RateLimitError):
            return ProviderRateLimitError(
                "Groq rate limit or quota was exceeded.",
                provider="groq",
            )

        if isinstance(error, groq.APITimeoutError):
            return ProviderTimeoutError(
                "Groq request timed out.",
                provider="groq",
            )

        if isinstance(error, groq.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to Groq.",
                provider="groq",
            )

        if isinstance(error, groq.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"Groq service returned status "
                    f"{status_code}.",
                    provider="groq",
                )

            return ProviderRequestError(
                f"Groq request failed with status "
                f"{status_code}.",
                provider="groq",
            )

        return ProviderError(
            f"Unexpected Groq provider error: {error}",
            provider="groq",
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
        """Generate a complete response with Groq."""

        config = settings.ai.groq

        request: dict[str, Any] = {
            "model": model or config.model,
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
                "Groq returned no choices.",
                provider=self.name,
            )

        message = response.choices[0].message
        content = message.content or ""

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=response.model,
            usage=self._usage(response),
            finish_reason=response.choices[0].finish_reason,
            request_id=getattr(
                response,
                "_request_id",
                None,
            ),
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
        """Stream a Groq response."""

        config = settings.ai.groq

        request: dict[str, Any] = {
            "model": model or config.model,
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
                        or config.model
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
        """Groq does not provide the embedding capability used here."""

        raise ProviderRequestError(
            "Groq does not provide an embedding API "
            "for this provider abstraction.",
            provider=self.name,
        )

    async def health_check(self) -> bool:
        """Check whether the configured Groq model is available."""

        config = settings.ai.groq

        try:
            await self._client.models.retrieve(
                config.model,
            )
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the Groq client."""

        await self._client.close()