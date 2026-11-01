"""Tests for runtime results."""

from uuid import uuid4

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.result import RuntimeResult
from app.runtime.types import RuntimeTokenUsage


def test_runtime_result_defaults() -> None:
    """RuntimeResult should provide sensible defaults."""

    result = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
    )

    assert result.output is None
    assert result.error is None
    assert result.metadata == {}
    assert isinstance(
        result.token_usage,
        RuntimeTokenUsage,
    )


def test_runtime_result_accepts_output() -> None:
    """RuntimeResult should store successful output."""

    result = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
        output="Hello!",
    )

    assert result.output == "Hello!"
    assert result.success is True


def test_runtime_result_accepts_error() -> None:
    """RuntimeResult should store execution errors."""

    result = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.FAILED,
        error="Provider failed",
    )

    assert result.error == "Provider failed"
    assert result.success is False


def test_runtime_result_success_only_for_completed() -> None:
    """Only completed executions should be successful."""

    statuses = (
        RuntimeStatus.IDLE,
        RuntimeStatus.RUNNING,
        RuntimeStatus.FAILED,
        RuntimeStatus.CANCELLED,
    )

    for status in statuses:
        result = RuntimeResult(
            request_id=uuid4(),
            workflow=Workflow.CHAT,
            status=status,
        )

        assert result.success is False


def test_runtime_result_accepts_metadata() -> None:
    """RuntimeResult should store metadata."""

    result = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        status=RuntimeStatus.COMPLETED,
        metadata={
            "provider": "openai",
            "streaming": True,
        },
    )

    assert result.metadata == {
        "provider": "openai",
        "streaming": True,
    }


def test_runtime_result_accepts_token_usage() -> None:
    """RuntimeResult should store token usage."""

    usage = RuntimeTokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    result = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
        token_usage=usage,
    )

    assert result.token_usage is usage
    assert result.token_usage.total_tokens == 150


def test_runtime_result_has_independent_mutable_defaults() -> None:
    """Mutable defaults should not be shared."""

    first = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
    )

    second = RuntimeResult(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.COMPLETED,
    )

    first.metadata["key"] = "value"

    assert second.metadata == {}