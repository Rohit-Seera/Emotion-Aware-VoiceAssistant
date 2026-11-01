"""Tests for runtime events."""

from datetime import datetime, timezone
from uuid import uuid4

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.events import (
    RuntimeCancelledEvent,
    RuntimeCompletedEvent,
    RuntimeEvent,
    RuntimeFailedEvent,
    RuntimeStartedEvent,
)


def test_runtime_event_defaults() -> None:
    """RuntimeEvent should provide sensible defaults."""

    request_id = uuid4()

    event = RuntimeEvent(
        request_id=request_id,
        workflow=Workflow.CHAT,
    )

    assert event.request_id == request_id
    assert event.workflow is Workflow.CHAT
    assert isinstance(event.timestamp, datetime)
    assert event.timestamp.tzinfo == timezone.utc
    assert event.metadata == {}


def test_runtime_event_accepts_metadata() -> None:
    """RuntimeEvent should accept metadata."""

    event = RuntimeEvent(
        request_id=uuid4(),
        workflow=Workflow.VOICE,
        metadata={
            "source": "microphone",
        },
    )

    assert event.metadata == {
        "source": "microphone",
    }


def test_runtime_started_event() -> None:
    """Started event should use running status."""

    event = RuntimeStartedEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    assert isinstance(event, RuntimeEvent)
    assert event.status is RuntimeStatus.RUNNING


def test_runtime_completed_event() -> None:
    """Completed event should use completed status."""

    event = RuntimeCompletedEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    assert isinstance(event, RuntimeEvent)
    assert event.status is RuntimeStatus.COMPLETED


def test_runtime_failed_event() -> None:
    """Failed event should store the error."""

    event = RuntimeFailedEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
        error="Provider failed",
    )

    assert isinstance(event, RuntimeEvent)
    assert event.status is RuntimeStatus.FAILED
    assert event.error == "Provider failed"


def test_runtime_cancelled_event() -> None:
    """Cancelled event should use cancelled status."""

    event = RuntimeCancelledEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    assert isinstance(event, RuntimeEvent)
    assert event.status is RuntimeStatus.CANCELLED


def test_event_timestamps_are_independent() -> None:
    """Each event should receive its own timestamp."""

    first = RuntimeEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    second = RuntimeEvent(
        request_id=uuid4(),
        workflow=Workflow.CHAT,
    )

    assert first.metadata is not second.metadata
    assert first.timestamp is not second.timestamp