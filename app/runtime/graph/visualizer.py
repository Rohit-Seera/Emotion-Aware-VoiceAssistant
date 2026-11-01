"""Runtime graph visualization utilities."""

from __future__ import annotations

from typing import Any


class RuntimeGraphVisualizer:
    """Provide inspection and visualization helpers for graphs."""

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @property
    def graph(self) -> Any:
        """Return the wrapped graph."""

        return self._graph

    def get_graph(self) -> Any:
        """Return the underlying graph representation."""

        getter = getattr(
            self._graph,
            "get_graph",
            None,
        )

        if getter is None:
            raise TypeError(
                "Graph does not expose get_graph()."
            )

        return getter()

    def draw_mermaid(self) -> str:
        """Return a Mermaid representation of the graph."""

        graph = self.get_graph()

        draw_mermaid = getattr(
            graph,
            "draw_mermaid",
            None,
        )

        if draw_mermaid is None:
            raise TypeError(
                "Graph does not support Mermaid visualization."
            )

        return draw_mermaid()