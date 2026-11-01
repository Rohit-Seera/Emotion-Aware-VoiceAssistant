"""Runtime execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.types import RuntimeTokenUsage


@dataclass(slots=True)
class RuntimeResult:
    """Final result returned by a runtime execution."""

    request_id: UUID
    workflow: Workflow
    status: RuntimeStatus

    output: str | None = None
    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    token_usage: RuntimeTokenUsage = field(
        default_factory=RuntimeTokenUsage,
    )

    @property
    def success(self) -> bool:
        """Return whether the execution completed successfully."""

        return self.status is RuntimeStatus.COMPLETED