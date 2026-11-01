"""Tests for memory summarization."""

from unittest.mock import AsyncMock

import pytest

from app.engines.memory.summarizer import MemorySummarizer


@pytest.mark.asyncio
async def test_summarize_calls_generator() -> None:
    """Summarizer should pass content to the configured generator."""

    generator = AsyncMock(
        return_value="User likes Python and AI."
    )

    summarizer = MemorySummarizer(
        generator=generator,
    )

    result = await summarizer.summarize(
        "The user said that they really like Python and AI."
    )

    assert result == "User likes Python and AI."

    generator.assert_awaited_once_with(
        "The user said that they really like Python and AI."
    )


@pytest.mark.asyncio
async def test_summarize_strips_result() -> None:
    """Summarizer should strip whitespace from generated summaries."""

    generator = AsyncMock(
        return_value="  User likes Python.  \n"
    )

    summarizer = MemorySummarizer(
        generator=generator,
    )

    result = await summarizer.summarize(
        "The user likes Python."
    )

    assert result == "User likes Python."


@pytest.mark.asyncio
async def test_summarize_rejects_empty_content() -> None:
    """Summarizer should reject empty content."""

    generator = AsyncMock()

    summarizer = MemorySummarizer(
        generator=generator,
    )

    with pytest.raises(
        ValueError,
        match="Content cannot be empty",
    ):
        await summarizer.summarize("   ")

    generator.assert_not_awaited()


@pytest.mark.asyncio
async def test_summarize_requires_provider() -> None:
    """Summarizer should require a configured provider."""

    summarizer = MemorySummarizer()

    with pytest.raises(
        RuntimeError,
        match="Summarization provider has not been configured",
    ):
        await summarizer.summarize(
            "Some conversation content."
        )


@pytest.mark.asyncio
async def test_summarize_propagates_generator_error() -> None:
    """Summarizer should propagate provider errors."""

    generator = AsyncMock(
        side_effect=RuntimeError("Provider failed")
    )

    summarizer = MemorySummarizer(
        generator=generator,
    )

    with pytest.raises(
        RuntimeError,
        match="Provider failed",
    ):
        await summarizer.summarize(
            "Some conversation content."
        )

    generator.assert_awaited_once_with(
        "Some conversation content."
    )