"""OpenAI provider implementation."""

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


class OpenAIProvider(BaseProvider):
    """OpenAI implementation of the common provider interface."""

    name = "openai"

    def __init__(self) -> None:
        config = settings.ai.openai

        if not config.enabled:
            raise ProviderConfigurationError(
                "OpenAI provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "OpenAI API key is not configured.",
                provider=self.name,
            )

        client_kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self._client = AsyncOpenAI(**client_kwargs)

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert provider usage information."""

        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        prompt_tokens = getattr(usage, "input_tokens", 0)
        completion_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        return ProviderUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _map_error(error: Exception) -> ProviderError:
        """Convert an OpenAI exception into a common provider error."""

        if isinstance(error, openai.AuthenticationError):
            return ProviderAuthenticationError(
                "OpenAI authentication failed.",
                provider="openai",
            )

        if isinstance(error, openai.RateLimitError):
            retry_after = None

            response = getattr(error, "response", None)

            if response is not None:
                header = response.headers.get("retry-after")

                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        retry_after = None

            return ProviderRateLimitError(
                "OpenAI rate limit exceeded.",
                provider="openai",
                retry_after=retry_after,
            )

        if isinstance(error, openai.APITimeoutError):
            return ProviderTimeoutError(
                "OpenAI request timed out.",
                provider="openai",
            )

        if isinstance(error, openai.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to OpenAI.",
                provider="openai",
            )

        if isinstance(error, openai.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"OpenAI service returned status {status_code}.",
                    provider="openai",
                )

            return ProviderRequestError(
                f"OpenAI request failed with status {status_code}.",
                provider="openai",
            )

        return ProviderError(
            "Unexpected OpenAI provider error.",
            provider="openai",
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
        """Generate a complete response."""

        config = settings.ai.openai

        request: dict[str, Any] = {
            "model": model or config.model,
            "input": list(messages),
            "temperature": (
                temperature
                if temperature is not None
                else config.temperature
            ),
            "max_output_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
        }

        request.update(kwargs)

        try:
            response = await self._client.responses.create(**request)
        except Exception as error:
            raise self._map_error(error) from error

        content = getattr(response, "output_text", None)

        if content is None:
            raise ProviderResponseError(
                "OpenAI returned no output text.",
                provider=self.name,
            )

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=response.model,
            usage=self._usage(response),
            request_id=getattr(response, "_request_id", None),
            metadata={
                "response_id": getattr(response, "id", None),
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
        """Stream response text incrementally."""

        config = settings.ai.openai

        request: dict[str, Any] = {
            "model": model or config.model,
            "input": list(messages),
            "temperature": (
                temperature
                if temperature is not None
                else config.temperature
            ),
            "max_output_tokens": (
                max_tokens
                if max_tokens is not None
                else config.max_tokens
            ),
            "stream": True,
        }

        request.update(kwargs)

        try:
            stream = await self._client.responses.create(**request)

            async for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.output_text.delta":
                    yield ProviderChunk(
                        content=event.delta,
                        provider=self.name,
                        model=model or config.model,
                    )

                elif event_type == "response.output_text.done":
                    yield ProviderChunk(
                        content="",
                        provider=self.name,
                        model=model or config.model,
                        metadata={
                            "text_completed": True,
                        },
                    )

                elif event_type == "response.completed":
                    response = getattr(event, "response", None)

                    yield ProviderChunk(
                        content="",
                        provider=self.name,
                        model=getattr(
                            response,
                            "model",
                            model or config.model,
                        ),
                        request_id=getattr(
                            response,
                            "_request_id",
                            None,
                        ),
                        finish_reason="completed",
                        metadata={
                            "response_id": getattr(
                                response,
                                "id",
                                None,
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
        """Generate embeddings for one or more texts."""

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                provider=self.name,
                model=model or "text-embedding-3-small",
            )

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Embedding input cannot contain empty text."
            )

        embedding_model = model or "text-embedding-3-small"

        request: dict[str, Any] = {
            "model": embedding_model,
            "input": list(texts),
            "dimensions": settings.memory.embedding_dimension,
        }

        request.update(kwargs)

        try:
            response = await self._client.embeddings.create(**request)
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
        """Check whether the configured OpenAI model is available."""

        config = settings.ai.openai

        try:
            await self._client.models.retrieve(config.model)
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the OpenAI client."""

        await self._client.close()