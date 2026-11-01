"""Tests for the model-router runtime node."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.nodes.model_router import (
    ModelRoute,
    ModelRouterNode,
)
from app.runtime.state import RuntimeState


@pytest.fixture
def state() -> RuntimeState:
    """Return a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello Weings.",
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
    )


def test_route_uses_explicit_configuration(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Explicit router configuration should take priority."""

    node = ModelRouterNode(
        provider="groq",
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2048,
    )

    route = node.route(
        state,
        context,
    )

    assert route == ModelRoute(
        provider="groq",
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        max_tokens=2048,
    )


def test_route_uses_context_configuration(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Context configuration should be used when provided."""

    context.metadata.update(
        {
            "provider": "openai",
            "model": "gpt-5",
            "temperature": 0.2,
            "max_tokens": 1000,
        }
    )

    node = ModelRouterNode()

    route = node.route(
        state,
        context,
    )

    assert route.provider == "openai"
    assert route.model == "gpt-5"
    assert route.temperature == 0.2
    assert route.max_tokens == 1000


def test_explicit_configuration_overrides_context(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """Explicit configuration should override context metadata."""

    context.metadata.update(
        {
            "provider": "openai",
            "model": "gpt-5",
            "temperature": 0.9,
            "max_tokens": 500,
        }
    )

    node = ModelRouterNode(
        provider="groq",
        model="test-model",
        temperature=0.3,
        max_tokens=2000,
    )

    route = node.route(
        state,
        context,
    )

    assert route.provider == "groq"
    assert route.model == "test-model"
    assert route.temperature == 0.3
    assert route.max_tokens == 2000


def test_route_returns_model_route(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """route should return a ModelRoute instance."""

    node = ModelRouterNode(
        provider="openai",
        model="gpt-5",
    )

    result = node.route(
        state,
        context,
    )

    assert isinstance(
        result,
        ModelRoute,
    )


@pytest.mark.asyncio
async def test_execute_stores_route(
    state: RuntimeState,
    context: RuntimeContext,
) -> None:
    """execute should store the selected route."""

    node = ModelRouterNode(
        provider="openai",
        model="gpt-5",
        temperature=0.5,
        max_tokens=3000,
    )

    result = await node.execute(
        state,
        context,
    )

    assert result is state

    assert context.metadata["model_route"] == {
        "provider": "openai",
        "model": "gpt-5",
        "temperature": 0.5,
        "max_tokens": 3000,
    }