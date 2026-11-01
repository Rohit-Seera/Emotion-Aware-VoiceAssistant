"""Tests for runtime graph visualization."""

from unittest.mock import MagicMock

import pytest

from app.runtime.graph.visualizer import RuntimeGraphVisualizer


def test_visualizer_stores_graph() -> None:
    """Visualizer should store the graph."""

    graph = MagicMock()

    visualizer = RuntimeGraphVisualizer(graph)

    assert visualizer.graph is graph


def test_get_graph_returns_graph_representation() -> None:
    """get_graph should return the graph representation."""

    representation = MagicMock()

    graph = MagicMock()
    graph.get_graph.return_value = representation

    visualizer = RuntimeGraphVisualizer(graph)

    result = visualizer.get_graph()

    assert result is representation
    graph.get_graph.assert_called_once_with()


def test_get_graph_rejects_graph_without_get_graph() -> None:
    """get_graph should reject unsupported graphs."""

    graph = object()

    visualizer = RuntimeGraphVisualizer(graph)

    with pytest.raises(
        TypeError,
        match=r"Graph does not expose get_graph",
    ):
        visualizer.get_graph()


def test_draw_mermaid_returns_mermaid() -> None:
    """draw_mermaid should return Mermaid text."""

    graph_representation = MagicMock()
    graph_representation.draw_mermaid.return_value = (
        "graph TD\nSTART --> node\nnode --> END"
    )

    graph = MagicMock()
    graph.get_graph.return_value = graph_representation

    visualizer = RuntimeGraphVisualizer(graph)

    result = visualizer.draw_mermaid()

    assert result == (
        "graph TD\nSTART --> node\nnode --> END"
    )


def test_draw_mermaid_rejects_unsupported_graph() -> None:
    """draw_mermaid should reject graphs without Mermaid support."""

    graph_representation = object()

    graph = MagicMock()
    graph.get_graph.return_value = graph_representation

    visualizer = RuntimeGraphVisualizer(graph)

    with pytest.raises(
        TypeError,
        match=r"Graph does not support Mermaid visualization",
    ):
        visualizer.draw_mermaid()