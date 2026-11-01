"""Adapter between runtime state and graph state."""

from __future__ import annotations

from typing import Any

from app.core.enums import RuntimeStatus, Workflow
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


class RuntimeStateAdapter:
    """Convert RuntimeState to and from graph-compatible state."""

    @staticmethod
    def to_dict(
        state: RuntimeState,
    ) -> dict[str, Any]:
        """Convert RuntimeState into a dictionary."""

        return {
            "request_id": state.request_id,
            "workflow": state.workflow,
            "input": state.input,
            "session_id": state.session_id,
            "status": state.status,
            "output": state.output,
            "error": state.error,
            "memories": list(state.memories),
            "metadata": dict(state.metadata),
            "token_usage": {
                "prompt_tokens": (
                    state.token_usage.prompt_tokens
                ),
                "completion_tokens": (
                    state.token_usage.completion_tokens
                ),
                "total_tokens": (
                    state.token_usage.total_tokens
                ),
            },
        }

    @staticmethod
    def from_dict(
        data: dict[str, Any],
    ) -> RuntimeState:
        """Build RuntimeState from a dictionary."""

        token_usage = data.get(
            "token_usage",
            {},
        )

        return RuntimeState(
            request_id=data["request_id"],
            workflow=(
                data["workflow"]
                if isinstance(data["workflow"], Workflow)
                else Workflow(data["workflow"])
            ),
            input=data["input"],
            session_id=data.get("session_id"),
            status=(
                data["status"]
                if isinstance(data["status"], RuntimeStatus)
                else RuntimeStatus(data["status"])
            ),
            output=data.get("output"),
            error=data.get("error"),
            memories=list(
                data.get("memories", []),
            ),
            metadata=dict(
                data.get("metadata", {}),
            ),
            token_usage=RuntimeTokenUsage(
                prompt_tokens=int(
                    token_usage.get(
                        "prompt_tokens",
                        0,
                    )
                ),
                completion_tokens=int(
                    token_usage.get(
                        "completion_tokens",
                        0,
                    )
                ),
                total_tokens=int(
                    token_usage.get(
                        "total_tokens",
                        0,
                    )
                ),
            ),
        )