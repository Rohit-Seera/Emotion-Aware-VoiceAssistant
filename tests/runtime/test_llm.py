"""Tests for the LLM runtime node."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.providers.base.types import ProviderResponse, ProviderUsage
from app.runtime.context import RuntimeContext
from app.runtime.nodes.llm import LLMNode
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello.",
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
            "prompt": "You are Weings. Respond warmly.",
            "model_route": {
                "provider": "openai",
                "model": "gpt-5",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        },
    )


def make_response() -> ProviderResponse:
    """Create a sample provider response."""

    return ProviderResponse(
        content="Hello! How can I help you?",
        provider="openai",
        model="gpt-5",
        usage=ProviderUsage(
            prompt_tokens=20,
            completion_tokens=8,
            total_tokens=28,
        ),
        finish_reason="stop",
        request_id="provider-request-123",
    )


@pytest.mark.asyncio
async def test_generates_response(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should generate a response."""

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=make_response(),
    )

    node = LLMNode(manager)

    result = await node.execute(
        state,
        context,
    )

    assert result is state
    assert state.output == (
        "Hello! How can I help you?"
    )
    assert context.metadata["llm_provider"] == "openai"
    assert context.metadata["llm_model"] == "gpt-5"

    manager.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_passes_prompt_and_model_route(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should pass the prompt and selected route."""

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=make_response(),
    )

    node = LLMNode(manager)

    await node.execute(
        state,
        context,
    )

    kwargs = manager.generate.await_args.kwargs
    args = manager.generate.await_args.args

    assert args[0] == [
        {
            "role": "user",
            "content": (
                "You are Weings. Respond warmly."
            ),
        },
    ]

    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt-5"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_maps_token_usage(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should map provider token usage to runtime state."""

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=make_response(),
    )

    node = LLMNode(manager)

    await node.execute(
        state,
        context,
    )

    assert state.token_usage.prompt_tokens == 20
    assert state.token_usage.completion_tokens == 8
    assert state.token_usage.total_tokens == 28


@pytest.mark.asyncio
async def test_stores_provider_response(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should store the provider response in context."""

    response = make_response()

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=response,
    )

    node = LLMNode(manager)

    await node.execute(
        state,
        context,
    )

    assert context.metadata["provider_response"] is response


@pytest.mark.asyncio
async def test_handles_missing_usage(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should handle providers without usage information."""

    response = ProviderResponse(
        content="Hello!",
        provider="openai",
        model="gpt-5",
    )

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=response,
    )

    node = LLMNode(manager)

    result = await node.execute(
        state,
        context,
    )

    assert result.output == "Hello!"
    assert result.token_usage.total_tokens == 0


@pytest.mark.asyncio
async def test_uses_context_provider_manager(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should use the manager from runtime context."""

    manager = MagicMock()
    manager.generate = AsyncMock(
        return_value=make_response(),
    )

    context.services["provider_manager"] = manager

    node = LLMNode()

    result = await node.execute(
        state,
        context,
    )

    assert result.output == (
        "Hello! How can I help you?"
    )

    manager.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_provider_manager_raises(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Node should fail when no provider manager is configured."""

    node = LLMNode()

    with pytest.raises(
        RuntimeError,
        match="Provider manager has not been configured",
    ):
        await node.execute(
            state,
            context,
        )


@pytest.mark.asyncio
async def test_missing_prompt_raises(
    state: RuntimeState,
) -> None:
    """Node should fail when no prompt has been built."""

    context = RuntimeContext(
        request_id=state.request_id,
        workflow=state.workflow,
        state=state,
        metadata={
            "model_route": {
                "provider": "openai",
                "model": "gpt-5",
            },
        },
    )

    manager = MagicMock()
    manager.generate = AsyncMock()

    node = LLMNode(manager)

    with pytest.raises(
        RuntimeError,
        match="Runtime prompt has not been configured",
    ):
        await node.execute(
            state,
            context,
        )

    manager.generate.assert_not_awaited()
