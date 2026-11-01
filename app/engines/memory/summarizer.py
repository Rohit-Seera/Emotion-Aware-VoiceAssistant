"""Memory summarization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable


class MemorySummarizer:
    """Generate concise summaries of memory content."""

    def __init__(
        self,
        generator: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._generator = generator

    async def summarize(
        self,
        content: str,
    ) -> str:
        """Generate a concise summary."""

        if not content.strip():
            raise ValueError("Content cannot be empty.")

        if self._generator is None:
            raise RuntimeError(
                "Summarization provider has not been configured."
            )

        summary = await self._generator(content)

        return summary.strip()