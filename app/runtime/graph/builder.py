"""LangGraph workflow builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.runtime.state import RuntimeState


class RuntimeGraphBuilder:
    """Build a LangGraph StateGraph for the runtime."""

    def __init__(self) -> None:
        self._builder = StateGraph(dict)

    def add_node(
        self,
        name: str,
        node: Callable[..., Any],
    ) -> RuntimeGraphBuilder:
        """Add a node to the graph."""

        if not name.strip():
            raise ValueError(
                "Node name cannot be empty."
            )

        self._builder.add_node(
            name,
            node,
        )

        return self

    def add_edge(
        self,
        source: str,
        target: str,
    ) -> RuntimeGraphBuilder:
        """Add a directed edge between nodes."""

        self._builder.add_edge(
            source,
            target,
        )

        return self

    def add_start_edge(
        self,
        target: str,
    ) -> RuntimeGraphBuilder:
        """Connect the graph START node to a node."""

        self._builder.add_edge(
            START,
            target,
        )

        return self

    def add_end_edge(
        self,
        source: str,
    ) -> RuntimeGraphBuilder:
        """Connect a node to the graph END."""

        self._builder.add_edge(
            source,
            END,
        )

        return self

    def build(self):
        """Return the uncompiled StateGraph builder."""

        return self._builder