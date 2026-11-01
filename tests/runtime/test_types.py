"""Tests for runtime types."""

from uuid import uuid4

import pytest

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.types import (
    RuntimeExecution,
    RuntimeMetadata,
    RuntimeRequest,
    RuntimeTokenUsage,
)


def test_runtime_request_defaults() -> None:
    """RuntimeRequest should provide sensible defaults."""

    request = RuntimeRequest(
        workflow=Workflow.CHAT,
        input="Hello",
    )

    assert request.workflow is Workflow.CHAT
    assert request.input == "Hello"
    assert request.session_id is None
    assert request.request_id is not None
    assert request.metadata == {}


def test_runtime_request_accepts_session_and_metadata() -> None:
    """RuntimeRequest should accept session and metadata."""

    session_id = uuid4()

    request = RuntimeRequest(
        workflow=Workflow.VOICE,
        input="Hello Weings",
        session_id=session_id,
        metadata={"source": "voice"},
    )

    assert request.workflow is Workflow.VOICE
    assert request.session_id == session_id
    assert request.metadata == {
        "source": "voice",
    }


def test_runtime_execution_defaults() -> None:
    """RuntimeExecution should start in the idle state."""

    request_id = uuid4()

    execution = RuntimeExecution(
        request_id=request_id,
        workflow=Workflow.CHAT,
    )

    assert execution.request_id == request_id
    assert execution.workflow is Workflow.CHAT
    assert execution.status is RuntimeStatus.IDLE
    assert execution.started_at is None
    assert execution.completed_at is None
    assert execution.error is None


def test_runtime_execution_tracks_error() -> None:
    """RuntimeExecution should store execution errors."""

    execution = RuntimeExecution(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        status=RuntimeStatus.FAILED,
        error="Provider failed",
    )

    assert execution.status is RuntimeStatus.FAILED
    assert execution.error == "Provider failed"


def test_runtime_token_usage_defaults() -> None:
    """Token usage should default to zero."""

    usage = RuntimeTokenUsage()

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_runtime_token_usage_accepts_values() -> None:
    """Token usage should accept token counts."""

    usage = RuntimeTokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150


def test_runtime_metadata_defaults() -> None:
    """Runtime metadata should default to an empty dictionary."""

    metadata = RuntimeMetadata()

    assert metadata.values == {}


def test_runtime_metadata_accepts_values() -> None:
    """Runtime metadata should accept arbitrary values."""

    metadata = RuntimeMetadata(
        values={
            "provider": "openai",
            "streaming": True,
        },
    )

    assert metadata.values == {
        "provider": "openai",
        "streaming": True,
    }