"""Gemini provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from google import genai
from google.genai import types

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


class GeminiProvider(BaseProvider):
    """Google Gemini implementation of the provider interface."""

    name = "gemini"

    def __init__(self) -> None:
        config = settings.ai.gemini

        if not config.enabled:
            raise ProviderConfigurationError(
                "Gemini provider is disabled.",
                provider=self.name,
            )

        if not config.api_key:
            raise ProviderConfigurationError(
                "Gemini API key is not configured.",
                provider=self.name,
            )

        self._client = genai.Client(
            api_key=config.api_key,
        )

    @staticmethod
    def _usage(response: Any) -> ProviderUsage | None:
        """Convert Gemini usage metadata."""

        usage = getattr(response, "usage_metadata", None)

        if usage is None:
            return None

        prompt_tokens = getattr(
            usage,
            "prompt_token_count",
            0,
        )

        completion_tokens = getattr(
            usage,
            "candidates_token_count",
            0,
        )

        total_tokens = getattr(
            usage,
            "total_token_count",
            0,
        )

        return ProviderUsage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _prepare_messages(
        messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert common messages into Gemini-compatible content."""

        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                parts = content
            else:
                parts = [{"text": str(content)}]

            if role == "system":
                system_parts = [
                    part.get("text", "")
                    for part in parts
                    if isinstance(part, dict)
                ]

                system_instruction = "\n".join(
                    part for part in system_parts if part
                )

                continue

            if role not in {"user", "model"}:
                role = "user"

            contents.append(
                {
                    "role": role,
                    "parts": parts,
                }
            )

        return contents, system_instruction

    @staticmethod
    def _map_error(error: Exception) -> ProviderError:
        """Convert Gemini SDK errors into common provider errors."""

        error_name = type(error).__name__
        message = str(error)

        if "Authentication" in error_name:
            return ProviderAuthenticationError(
                "Gemini authentication failed.",
                provider="gemini",
            )

        if "Permission" in error_name:
            return ProviderAuthenticationError(
                "Gemini authentication or permission failed.",
                provider="gemini",
            )

        if "ResourceExhausted" in error_name:
            return ProviderRateLimitError(
                "Gemini rate limit or quota was exceeded.",
                provider="gemini",
            )

        if "Timeout" in error_name:
            return ProviderTimeoutError(
                "Gemini request timed out.",
                provider="gemini",
            )

        if "Connection" in error_name:
            return ProviderUnavailableError(
                "Unable to connect to Gemini.",
                provider="gemini",
            )

        if "Unavailable" in error_name:
            return ProviderUnavailableError(
                "Gemini service is temporarily unavailable.",
                provider="gemini",
            )

        if "InvalidArgument" in error_name:
            return ProviderRequestError(
                f"Gemini rejected the request: {message}",
                provider="gemini",
            )

        return ProviderError(
            f"Unexpected Gemini provider error: {message}",
            provider="gemini",
        )

    def _build_config(
        self,
        *,
        temperature: float | None,
        max_tokens: int | None,
        system_instruction: str | None,
        kwargs: dict[str, Any],
    ) -> types.GenerateContentConfig:
        """Build Gemini generation configuration."""

        config = dict(kwargs)

        config.setdefault(
            "temperature",
            (
                temperature
                if temperature is not None
                else settings.ai.gemini.temperature
            ),
        )

        config.setdefault(
            "max_output_tokens",
            (
                max_tokens
                if max_tokens is not None
                else settings.ai.gemini.max_tokens
            ),
        )

        if system_instruction:
            config.setdefault(
                "system_instruction",
                system_instruction,
            )

        return types.GenerateContentConfig(**config)

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a complete Gemini response."""

        config = settings.ai.gemini

        contents, system_instruction = self._prepare_messages(
            messages
        )

        generation_config = self._build_config(
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            kwargs=kwargs,
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=model or config.model,
                contents=contents,
                config=generation_config,
            )
        except Exception as error:
            raise self._map_error(error) from error

        content = getattr(response, "text", None)

        if not content:
            raise ProviderResponseError(
                "Gemini returned no text output.",
                provider=self.name,
            )

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=getattr(
                response,
                "model_version",
                model or config.model,
            ),
            usage=self._usage(response),
            metadata={},
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
        """Stream Gemini response chunks."""

        config = settings.ai.gemini

        contents, system_instruction = self._prepare_messages(
            messages
        )

        generation_config = self._build_config(
            temperature=temperature,
            max_tokens=max_tokens,
            system_instruction=system_instruction,
            kwargs=kwargs,
        )

        try:
            stream = (
                await self._client.aio.models.generate_content_stream(
                    model=model or config.model,
                    contents=contents,
                    config=generation_config,
                )
            )

            async for chunk in stream:
                text = getattr(chunk, "text", None)

                if not text:
                    continue

                yield ProviderChunk(
                    content=text,
                    provider=self.name,
                    model=getattr(
                        chunk,
                        "model_version",
                        model or config.model,
                    ),
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
        """Generate Gemini embeddings."""

        if not texts:
            return EmbeddingResult(
                embeddings=[],
                provider=self.name,
                model=model or "gemini-embedding-2",
            )

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Embedding input cannot contain empty text."
            )

        embedding_model = model or "gemini-embedding-2"

        config_kwargs = dict(kwargs)

        config_kwargs.setdefault(
            "output_dimensionality",
            settings.memory.embedding_dimension,
        )

        embedding_config = types.EmbedContentConfig(
            **config_kwargs
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=embedding_model,
                contents=list(texts),
                config=embedding_config,
            )
        except Exception as error:
            raise self._map_error(error) from error

        embeddings = [
            embedding.values
            for embedding in response.embeddings
            if embedding.values is not None
        ]

        return EmbeddingResult(
            embeddings=embeddings,
            provider=self.name,
            model=embedding_model,
        )

    async def health_check(self) -> bool:
        """Check whether Gemini is reachable."""

        config = settings.ai.gemini

        try:
            await self._client.aio.models.get(
                model=config.model,
            )
        except Exception:
            return False

        return True

    async def close(self) -> None:
        """Close the Gemini client."""

        await self._client.aio.aclose()