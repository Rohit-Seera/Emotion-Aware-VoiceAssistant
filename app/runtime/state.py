"""Runtime workflow state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.types import RuntimeTokenUsage


@dataclass(slots=True)
class RuntimeState:
    """State carried through a runtime workflow."""

    request_id: UUID
    workflow: Workflow
    input: str

    session_id: UUID | None = None

    status: RuntimeStatus = RuntimeStatus.IDLE

    output: str | None = None
    error: str | None = None

    memories: list[Any] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    token_usage: RuntimeTokenUsage = field(
        default_factory=RuntimeTokenUsage,
    )