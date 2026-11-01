"""Tests for the runtime state adapter."""

from uuid import uuid4

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.adapters.state_adapter import RuntimeStateAdapter
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


def make_state() -> RuntimeState:
    """Create a sample runtime state."""

    return RuntimeState(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        input="Hello",
        session_id=uuid4(),
        status=RuntimeStatus.RUNNING,
        output="Hi there!",
        metadata={
            "provider": "openai",
        },
        memories=[
            {"content": "User likes Python."},
        ],
        token_usage=RuntimeTokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def test_to_dict_returns_runtime_state_data() -> None:
    """Adapter should convert RuntimeState into a dictionary."""

    state = make_state()

    result = RuntimeStateAdapter.to_dict(state)

    assert result["request_id"] == state.request_id
    assert result["workflow"] == state.workflow
    assert result["input"] == state.input
    assert result["session_id"] == state.session_id
    assert result["status"] == state.status
    assert result["output"] == state.output
    assert result["memories"] == state.memories
    assert result["metadata"] == state.metadata


def test_to_dict_includes_token_usage() -> None:
    """Adapter should preserve token usage."""

    state = make_state()

    result = RuntimeStateAdapter.to_dict(state)

    assert result["token_usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_from_dict_reconstructs_state() -> None:
    """Adapter should reconstruct RuntimeState from a dictionary."""

    state = make_state()

    data = RuntimeStateAdapter.to_dict(state)

    result = RuntimeStateAdapter.from_dict(data)

    assert result.request_id == state.request_id
    assert result.workflow is state.workflow
    assert result.input == state.input
    assert result.session_id == state.session_id
    assert result.status is state.status
    assert result.output == state.output
    assert result.memories == state.memories
    assert result.metadata == state.metadata


def test_from_dict_reconstructs_token_usage() -> None:
    """Adapter should reconstruct token usage."""

    state = make_state()

    data = RuntimeStateAdapter.to_dict(state)

    result = RuntimeStateAdapter.from_dict(data)

    assert result.token_usage.prompt_tokens == 10
    assert result.token_usage.completion_tokens == 5
    assert result.token_usage.total_tokens == 15


def test_round_trip_preserves_state() -> None:
    """Converting to dict and back should preserve state."""

    state = make_state()

    result = RuntimeStateAdapter.from_dict(
        RuntimeStateAdapter.to_dict(state),
    )

    assert result == state