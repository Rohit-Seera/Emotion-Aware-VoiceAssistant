"""Runtime execution events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from app.core.enums import RuntimeStatus, Workflow


@dataclass(slots=True)
class RuntimeEvent:
    """Base event emitted during runtime execution."""

    request_id: UUID
    workflow: Workflow
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RuntimeStartedEvent(RuntimeEvent):
    """Event emitted when runtime execution starts."""

    status: RuntimeStatus = RuntimeStatus.RUNNING


@dataclass(slots=True)
class RuntimeCompletedEvent(RuntimeEvent):
    """Event emitted when runtime execution completes."""

    status: RuntimeStatus = RuntimeStatus.COMPLETED


@dataclass(slots=True)
class RuntimeFailedEvent(RuntimeEvent):
    """Event emitted when runtime execution fails."""

    status: RuntimeStatus = RuntimeStatus.FAILED
    error: str = ""


@dataclass(slots=True)
class RuntimeCancelledEvent(RuntimeEvent):
    """Event emitted when runtime execution is cancelled."""

    status: RuntimeStatus = RuntimeStatus.CANCELLED