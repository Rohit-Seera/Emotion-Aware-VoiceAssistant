"""Core runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from app.core.enums import RuntimeStatus, Workflow


@dataclass(slots=True)
class RuntimeRequest:
    """Input required to execute a workflow."""

    workflow: Workflow
    input: str
    session_id: UUID | None = None
    request_id: UUID = field(
        default_factory=uuid4,
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RuntimeExecution:
    """Track the lifecycle of a runtime execution."""

    request_id: UUID
    workflow: Workflow
    status: RuntimeStatus = RuntimeStatus.IDLE
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


@dataclass(slots=True)
class RuntimeTokenUsage:
    """Track token usage during an execution."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class RuntimeMetadata:
    """Additional metadata associated with execution."""

    values: dict[str, Any] = field(
        default_factory=dict,
    )