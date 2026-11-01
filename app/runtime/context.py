"""Runtime execution context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.core.enums import Workflow
from app.runtime.state import RuntimeState


@dataclass(slots=True)
class RuntimeContext:
    """Provide stable context and dependencies for an execution."""

    request_id: UUID
    workflow: Workflow

    state: RuntimeState | None = None

    services: dict[str, Any] = field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def get_service(
        self,
        name: str,
    ) -> Any:
        """Return a registered service."""

        if name not in self.services:
            raise KeyError(
                f"Runtime service '{name}' is not registered."
            )

        return self.services[name]

    def has_service(
        self,
        name: str,
    ) -> bool:
        """Check whether a service is registered."""

        return name in self.services