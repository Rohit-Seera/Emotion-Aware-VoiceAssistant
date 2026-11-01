"""Tests for runtime context."""

from uuid import uuid4

import pytest

from app.core.enums import Workflow
from app.runtime.context import RuntimeContext
from app.runtime.state import RuntimeState


def test_runtime_context_defaults() -> None:
    """RuntimeContext should provide sensible defaults."""

    request_id = uuid4()

    context = RuntimeContext(
        request_id=request_id,
        workflow=Workflow.CHAT,
    )

    assert context.request_id == request_id
    assert context.workflow is Workflow.CHAT
    assert context.state is None
    assert context.services == {}
    assert context.metadata == {}


def test_runtime_context_accepts_state() -> None:
    """RuntimeContext should accept runtime state."""

    request_id = uuid4()

    state = RuntimeState(
        request_id=request_id,
        workflow=Workflow.CHAT,
        input="Hello",
    )

    context = RuntimeContext(
        request_id=request_id,
        workflow=Workflow.CHAT,
        state=state,
    )

    assert context.state is state


def test_runtime_context_accepts_services() -> None:
    """RuntimeContext should store runtime services."""

    repository = object()
    provider = object()

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        services={
            "repository": repository,
            "provider": provider,
        },
    )

    assert context.services["repository"] is repository
    assert context.services["provider"] is provider


def test_get_service_returns_registered_service() -> None:
    """get_service should return a registered service."""

    provider = object()

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        services={
            "provider": provider,
        },
    )

    result = context.get_service("provider")

    assert result is provider


def test_get_service_raises_for_missing_service() -> None:
    """get_service should raise when a service is missing."""

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    with pytest.raises(
        KeyError,
        match="Runtime service 'provider' is not registered",
    ):
        context.get_service("provider")


def test_has_service_returns_true_when_registered() -> None:
    """has_service should return True for registered services."""

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        services={
            "provider": object(),
        },
    )

    assert context.has_service("provider") is True


def test_has_service_returns_false_when_missing() -> None:
    """has_service should return False for missing services."""

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    assert context.has_service("provider") is False


def test_runtime_context_accepts_metadata() -> None:
    """RuntimeContext should store execution metadata."""

    context = RuntimeContext(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        metadata={
            "streaming": True,
            "source": "voice",
        },
    )

    assert context.metadata == {
        "streaming": True,
        "source": "voice",
    }