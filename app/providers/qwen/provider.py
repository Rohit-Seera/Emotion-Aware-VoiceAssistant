"""Qwen provider implementation."""

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


class QwenProvider(BaseProvider):
    """Qwen implementation of the common provider interface."""

    name = "qwen"

    def __init__(self) -> None:
        config = settings.ai.qwen

        if not config.enabled:
            raise ProviderConfigurationError(
                "Qwen provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "Qwen API key is not configured.",
                provider=self.name,
            )

        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=(
                config.base_url
                or "https://dashscope.aliyuncs.com/"
                "compatible-mode/v1"
            ),
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert Qwen usage metadata."""

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
                "Qwen authentication failed.",
                provider="qwen",
            )

        if isinstance(error, openai.RateLimitError):
            return ProviderRateLimitError(
                "Qwen rate limit or quota was exceeded.",
                provider="qwen",
            )

        if isinstance(error, openai.APITimeoutError):
            return ProviderTimeoutError(
                "Qwen request timed out.",
                provider="qwen",
            )

        if isinstance(error, openai.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to Qwen.",
                provider="qwen",
            )

        if isinstance(error, openai.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"Qwen service returned status "
                    f"{status_code}.",
                    provider="qwen",
                )

            return ProviderRequestError(
                f"Qwen request failed with status "
                f"{status_code}.",
                provider="qwen",
            )

        return ProviderError(
            f"Unexpected Qwen provider error: {error}",
            provider="qwen",
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
        """Generate a complete Qwen response."""

        config = settings.ai.qwen

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
                "Qwen returned no choices.",
                provider=self.name,
            )

        choice = response.choices[0]
        message = choice.message

        content = message.content or ""

        reasoning_content = getattr(
            message,
            "reasoning_content",
            None,
        )

        return ProviderResponse(
            content=content,
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
                "reasoning_content": reasoning_content,
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
        """Stream a Qwen response."""

        config = settings.ai.qwen

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
                delta = choice.delta

                content = delta.content or ""
                reasoning_content = getattr(
                    delta,
                    "reasoning_content",
                    None,
                )

                if not content and not reasoning_content:
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
                    metadata={
                        "reasoning_content": (
                            reasoning_content
                        ),
                    },
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
        """Qwen embedding support is not assumed here."""

        raise ProviderRequestError(
            "Qwen embedding support is not configured "
            "for this provider abstraction.",
            provider=self.name,
        )

    async def health_check(self) -> bool:
        """Check whether the configured Qwen model is available."""

        config = settings.ai.qwen

        try:
            await self._client.models.retrieve(
                config.model,
            )
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the Qwen client."""

        await self._client.close()