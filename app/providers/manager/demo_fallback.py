"""Deterministic development fallback used only when real LLM providers fail."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.providers.base.provider import BaseProvider
from app.providers.base.types import EmbeddingResult, ProviderChunk, ProviderResponse


class DemoFallbackProvider(BaseProvider):
    """Provide a clearly-labelled local reply so the demo remains usable.

    This path is intentionally development-only. It is not a substitute for a
    research result and is reported in provider metadata/UI when selected.
    """

    name = "demo_fallback"

    @staticmethod
    def _extract(prompt: str, key: str) -> str:
        marker = f"{key}:"
        for line in prompt.splitlines():
            if line.startswith(marker):
                return line[len(marker):].strip()
        return ""

    def _reply(self, prompt: str) -> str:
        user = self._extract(prompt, "User")
        emotion = self._extract(prompt, "Emotion").split("(", 1)[0].strip().casefold()

        if "start the conversation" in user.casefold():
            return (
                "Hi, I’m Aiko. I can adapt my response to your emotional tone, "
                "and you can interrupt me anytime while I’m speaking."
            )

        if emotion in {"sad", "anxious", "fearful", "frustrated", "angry"}:
            return (
                "I hear you. Let’s take this one step at a time, and I’ll stay "
                "with you through it. Tell me what you want to work through."
            )

        if emotion == "happy":
            return (
                "That sounds great! I’m glad to hear it. Tell me what you’d "
                "like to do next."
            )

        text = user.strip()
        if not text:
            return "I’m here with you. What would you like to talk about?"
        return f"I heard you say, “{text}”. I’m ready to help you with that."

    async def generate(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        del model, temperature, max_tokens, kwargs
        prompt = str(messages[-1].get("content", "")) if messages else ""
        return ProviderResponse(
            content=self._reply(prompt),
            provider=self.name,
            model="local-demo-fallback",
            metadata={"demo_fallback": True},
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
        response = await self.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        words = response.content.split(" ")
        for index, word in enumerate(words):
            yield ProviderChunk(
                content=(word if index == 0 else f" {word}"),
                provider=self.name,
                model=response.model,
            )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        del texts, model, kwargs
        raise RuntimeError("Demo fallback does not provide embeddings.")
