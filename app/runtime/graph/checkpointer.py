"""LangGraph checkpoint management."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver


class RuntimeCheckpointer:
    """Manage the checkpointer used by runtime graphs."""

    def __init__(
        self,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        self._checkpointer = (
            checkpointer
            if checkpointer is not None
            else InMemorySaver()
        )

    @property
    def checkpointer(self) -> InMemorySaver:
        """Return the configured LangGraph checkpointer."""

        return self._checkpointer