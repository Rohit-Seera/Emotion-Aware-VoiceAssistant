"""Anthropic provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

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


class AnthropicProvider(BaseProvider):
    """Anthropic implementation of the common provider interface."""

    name = "anthropic"

    def __init__(self) -> None:
        config = settings.ai.anthropic

        if not config.enabled:
            raise ProviderConfigurationError(
                "Anthropic provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "Anthropic API key is not configured.",
                provider=self.name,
            )

        self._client = AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert Anthropic usage metadata."""

        usage = getattr(response, "usage", None)

        if usage is None:
            return None

        prompt_tokens = getattr(
            usage,
            "input_tokens",
            0,
        )

        completion_tokens = getattr(
            usage,
            "output_tokens",
            0,
        )

        return ProviderUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(
                (prompt_tokens or 0)
                + (completion_tokens or 0)
            ),
        )

    @staticmethod
    def _prepare_messages(
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert common messages into Anthropic format."""

        prepared: list[dict[str, Any]] = []
        system_parts: list[str] = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text")
                            if text:
                                system_parts.append(str(text))
                continue

            if role not in {"user", "assistant"}:
                role = "user"

            prepared.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        system_instruction = "\n\n".join(
            part for part in system_parts if part
        )

        return prepared, system_instruction or None

    @staticmethod
    def _map_error(error: Exception) -> ProviderError:
        """Convert Anthropic SDK errors to common provider errors."""

        if isinstance(error, anthropic.AuthenticationError):
            return ProviderAuthenticationError(
                "Anthropic authentication failed.",
                provider="anthropic",
            )

        if isinstance(error, anthropic.RateLimitError):
            return ProviderRateLimitError(
                "Anthropic rate limit exceeded.",
                provider="anthropic",
            )

        if isinstance(error, anthropic.APITimeoutError):
            return ProviderTimeoutError(
                "Anthropic request timed out.",
                provider="anthropic",
            )

        if isinstance(error, anthropic.APIConnectionError):
            return ProviderUnavailableError(
                "Unable to connect to Anthropic.",
                provider="anthropic",
            )

        if isinstance(error, anthropic.APIStatusError):
            status_code = error.status_code

            if status_code >= 500:
                return ProviderUnavailableError(
                    f"Anthropic service returned status "
                    f"{status_code}.",
                    provider="anthropic",
                )

            return ProviderRequestError(
                f"Anthropic request failed with status "
                f"{status_code}.",
                provider="anthropic",
            )

        return ProviderError(
            f"Unexpected Anthropic provider error: {error}",
            provider="anthropic",
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
        """Generate a complete Anthropic response."""

        config = settings.ai.anthropic

        prepared_messages, system_instruction = (
            self._prepare_messages(messages)
        )

        request: dict[str, Any] = {
            "model": model or config.model,
            "messages": prepared_messages,
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

        if system_instruction:
            request["system"] = system_instruction

        request.update(kwargs)

        try:
            response = await self._client.messages.create(
                **request
            )
        except Exception as error:
            raise self._map_error(error) from error

        text_parts: list[str] = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        content = "".join(text_parts)

        if not content:
            raise ProviderResponseError(
                "Anthropic returned no text output.",
                provider=self.name,
            )

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=response.model,
            usage=self._usage(response),
            finish_reason=response.stop_reason,
            request_id=getattr(
                response,
                "_request_id",
                None,
            ),
            metadata={
                "response_id": response.id,
                "stop_sequence": response.stop_sequence,
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
        """Stream Anthropic response text."""

        config = settings.ai.anthropic

        prepared_messages, system_instruction = (
            self._prepare_messages(messages)
        )

        request: dict[str, Any] = {
            "model": model or config.model,
            "messages": prepared_messages,
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

        if system_instruction:
            request["system"] = system_instruction

        request.update(kwargs)

        try:
            async with self._client.messages.stream(
                **request
            ) as stream:
                async for text in stream.text_stream:
                    if not text:
                        continue

                    yield ProviderChunk(
                        content=text,
                        provider=self.name,
                        model=model or config.model,
                        request_id=stream.request_id,
                    )

                final_message = await stream.get_final_message()

                yield ProviderChunk(
                    content="",
                    provider=self.name,
                    model=final_message.model,
                    request_id=stream.request_id,
                    finish_reason=final_message.stop_reason,
                    metadata={
                        "response_id": final_message.id,
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
        """Anthropic does not provide a native embeddings API."""

        raise ProviderRequestError(
            "Anthropic does not provide an embeddings API. "
            "Use a dedicated embedding provider.",
            provider=self.name,
        )

    async def health_check(self) -> bool:
        """Check whether Anthropic is reachable."""

        config = settings.ai.anthropic

        try:
            await self._client.models.retrieve(
                config.model,
            )
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the Anthropic client."""

        await self._client.close()