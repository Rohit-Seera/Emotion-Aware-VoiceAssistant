"""DeepSeek provider implementation."""

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


class DeepSeekProvider(BaseProvider):
    """DeepSeek implementation of the common provider interface."""

    name = "deepseek"

    def __init__(self) -> None:
        config = settings.ai.deepseek

        if not config.enabled:
            raise ProviderConfigurationError(
                "DeepSeek provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "DeepSeek API key is not configured.",
                provider=self.name,
            )

        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
            or "https://api.deepseek.com",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert DeepSeek usage metadata."""

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
        """Convert OpenAI-compatible errors to provider errors."""

        if isinstance(error, openai.AuthenticationError):
            return ProviderAuthenticationError(
                "DeepSeek authentication failed.",
                provider="deepseek",
            )

        if isinstance(error, openai.RateLimitError):
            return ProviderRateLimitError(
                "DeepSeek rate limit or quota was exceeded.",
                provider="deepseek",
            )

        if isinstance(error, openai.APITimeoutError):
            return ProviderTimeoutError(
                "DeepSeek request timed out.",
                provider="deepseek",
            )

        if isinstance(error, openai.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to DeepSeek.",
                provider="deepseek",
            )

        if isinstance(error, openai.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"DeepSeek service returned status "
                    f"{status_code}.",
                    provider="deepseek",
                )

            return ProviderRequestError(
                f"DeepSeek request failed with status "
                f"{status_code}.",
                provider="deepseek",
            )

        return ProviderError(
            f"Unexpected DeepSeek provider error: {error}",
            provider="deepseek",
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
        """Generate a complete DeepSeek response."""

        config = settings.ai.deepseek

        request: dict[str, Any] = {
            "model": model or config.model,
            "messages": list(messages),
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
        }

        if temperature is not None:
            request["temperature"] = temperature
        else:
            request["temperature"] = config.temperature

        request.update(kwargs)

        try:
            response = await self._client.chat.completions.create(
                **request
            )
        except Exception as error:
            raise self._map_error(error) from error

        if not response.choices:
            raise ProviderResponseError(
                "DeepSeek returned no choices.",
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
        """Stream a DeepSeek response."""

        config = settings.ai.deepseek

        request: dict[str, Any] = {
            "model": model or config.model,
            "messages": list(messages),
            "stream": True,
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
        }

        if temperature is not None:
            request["temperature"] = temperature
        else:
            request["temperature"] = config.temperature

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
        """DeepSeek does not provide embeddings through this API."""

        raise ProviderRequestError(
            "DeepSeek does not provide an embeddings API "
            "for this provider abstraction.",
            provider=self.name,
        )

    async def health_check(self) -> bool:
        """Check whether the configured DeepSeek model is available."""

        config = settings.ai.deepseek

        try:
            await self._client.models.retrieve(
                config.model,
            )
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the DeepSeek client."""

        await self._client.close()