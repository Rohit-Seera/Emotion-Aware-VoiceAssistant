"""Tests for the summarize runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.summarize import SummarizeNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Tell me about Python.",
    )


@pytest.fixture
def context(
    state: RuntimeState,
) -> RuntimeContext:
    """Return a sample runtime context."""

    return RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "content_to_summarize": (
                "Python is a programming language "
                "used for many different applications."
            ),
        },
    )


def make_summarizer() -> MagicMock:
    """Create a mocked summarization service."""

    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(
        return_value="Python is a versatile programming language.",
    )

    return summarizer


@pytest.mark.asyncio
async def test_summarizes_content(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should summarize the configured content."""

    summarizer = make_summarizer()

    node = SummarizeNode(summarizer)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["summary"] == (
        "Python is a versatile programming language."
    )

    summarizer.summarize.assert_awaited_once_with(
        "Python is a programming language "
        "used for many different applications.",
    )


@pytest.mark.asyncio
async def test_uses_context_summarizer(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the summarizer from runtime context."""

    summarizer = make_summarizer()

    context.services["memory_summarizer"] = summarizer

    node = SummarizeNode()

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert context.metadata["summary"] == (
        "Python is a versatile programming language."
    )

    summarizer.summarize.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_content_raises(
    state: RuntimeState,
) -> None:
    """Node should reject missing summarization content."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
    )

    summarizer = make_summarizer()

    node = SummarizeNode(summarizer)

    with pytest.raises(
        ValueError,
        match="Content to summarize must be a non-empty string",
    ):
        await node.execute(
            state,
            context,
        )

    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_content_raises(
    state: RuntimeState,
) -> None:
    """Node should reject whitespace-only content."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "content_to_summarize": "   ",
        },
    )

    summarizer = make_summarizer()

    node = SummarizeNode(summarizer)

    with pytest.raises(
        ValueError,
        match="Content to summarize must be a non-empty string",
    ):
        await node.execute(
            state,
            context,
        )

    summarizer.summarize.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_summarizer_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail without a summarization service."""

    node = SummarizeNode()

    with pytest.raises(
        RuntimeError,
        match="Memory summarizer has not been configured",
    ):
        await node.execute(
            state,
            context,
        )


@pytest.mark.asyncio
async def test_stores_summary(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should store the generated summary."""

    summarizer = make_summarizer()

    node = SummarizeNode(summarizer)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["summary"] == (
        "Python is a versatile programming language."
    )