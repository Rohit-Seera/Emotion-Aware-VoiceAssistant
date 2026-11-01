"""Summarize runtime content."""

from __future__ import annotations

from typing import Protocol

from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


class SummarizerProtocol(Protocol):
    """Contract for the summarization service."""

    async def summarize(
        self,
        content: str,
    ) -> str:
        """Generate a concise summary."""
        ...


class SummarizeNode:
    """Generate and store a summary for runtime content."""

    def __init__(
        self,
        summarizer: SummarizerProtocol | None = None,
    ) -> None:
        self._summarizer = summarizer

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState:
        """Summarize the configured runtime content."""

        content = context.metadata.get(
            "content_to_summarize",
        )

        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Content to summarize must be a non-empty string."
            )

        summarizer = self._summarizer

        if summarizer is None:
            if not context.has_service("memory_summarizer"):
                raise RuntimeError(
                    "Memory summarizer has not been configured."
                )

            summarizer = context.get_service(
                "memory_summarizer",
            )

        summary = await summarizer.summarize(
            content,
        )

        context.metadata["summary"] = summary

        return state