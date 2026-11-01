"""Tests for runtime state."""

from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


def test_runtime_state_defaults() -> None:
    """RuntimeState should provide sensible defaults."""

    request_id = uuid4()

    state = RuntimeState(
        request_id=request_id,
        workflow=Workflow.CHAT,
        input="Hello",
    )

    assert state.request_id == request_id
    assert state.workflow is Workflow.CHAT
    assert state.input == "Hello"

    assert state.session_id is None
    assert state.status is RuntimeStatus.IDLE
    assert state.output is None
    assert state.error is None

    assert state.memories == []
    assert state.metadata == {}

    assert isinstance(
        state.token_usage,
        RuntimeTokenUsage,
    )


def test_runtime_state_accepts_session() -> None:
    """RuntimeState should accept a session identifier."""

    session_id = uuid4()

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        input="Hello Weings",
        session_id=session_id,
    )

    assert state.session_id == session_id


def test_runtime_state_tracks_output() -> None:
    """RuntimeState should store workflow output."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        output="Hello! How can I help?",
        status=RuntimeStatus.COMPLETED,
    )

    assert state.output == "Hello! How can I help?"
    assert state.status is RuntimeStatus.COMPLETED


def test_runtime_state_tracks_error() -> None:
    """RuntimeState should store workflow errors."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        status=RuntimeStatus.FAILED,
        error="Provider failed",
    )

    assert state.status is RuntimeStatus.FAILED
    assert state.error == "Provider failed"


def test_runtime_state_accepts_memories() -> None:
    """RuntimeState should store retrieved memories."""

    memories = [
        {"content": "User likes Python."},
        {"content": "User works with AI."},
    ]

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="What do I like?",
        memories=memories,
    )

    assert state.memories == memories


def test_runtime_state_accepts_metadata() -> None:
    """RuntimeState should store execution metadata."""

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        metadata={
            "provider": "openai",
            "streaming": True,
        },
    )

    assert state.metadata == {
        "provider": "openai",
        "streaming": True,
    }


def test_runtime_state_accepts_token_usage() -> None:
    """RuntimeState should track token usage."""

    usage = RuntimeTokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    state = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        token_usage=usage,
    )

    assert state.token_usage is usage
    assert state.token_usage.total_tokens == 150


def test_runtime_state_instances_have_independent_defaults() -> None:
    """Mutable defaults should not be shared between states."""

    first = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="First",
    )

    second = RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Second",
    )

    first.memories.append("memory")
    first.metadata["key"] = "value"

    assert second.memories == []
    assert second.metadata == {}