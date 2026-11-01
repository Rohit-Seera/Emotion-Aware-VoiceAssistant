"""Tests for the runtime graph builder."""

from unittest.mock import MagicMock

import pytest
from langgraph.graph import END, START

from app.runtime.graph.builder import RuntimeGraphBuilder


def test_builder_creates_state_graph() -> None:
    """Builder should create a LangGraph StateGraph."""

    builder = RuntimeGraphBuilder()

    assert builder.build() is not None


def test_add_node_returns_builder() -> None:
    """add_node should support fluent chaining."""

    builder = RuntimeGraphBuilder()
    node = MagicMock()

    result = builder.add_node(
        "test",
        node,
    )

    assert result is builder


def test_add_empty_node_name_raises() -> None:
    """Empty node names should be rejected."""

    builder = RuntimeGraphBuilder()

    with pytest.raises(
        ValueError,
        match="Node name cannot be empty",
    ):
        builder.add_node(
            "   ",
            MagicMock(),
        )


def test_add_edge_returns_builder() -> None:
    """add_edge should support fluent chaining."""

    builder = RuntimeGraphBuilder()

    result = builder.add_edge(
        "first",
        "second",
    )

    assert result is builder


def test_add_start_edge_returns_builder() -> None:
    """add_start_edge should support fluent chaining."""

    builder = RuntimeGraphBuilder()

    result = builder.add_start_edge(
        "first",
    )

    assert result is builder


def test_add_end_edge_returns_builder() -> None:
    """add_end_edge should support fluent chaining."""

    builder = RuntimeGraphBuilder()

    result = builder.add_end_edge(
        "last",
    )

    assert result is builder


def test_builds_simple_graph() -> None:
    """Builder should create a valid simple graph."""

    builder = RuntimeGraphBuilder()

    async def first(state: dict) -> dict:
        return state

    async def last(state: dict) -> dict:
        return state

    builder.add_node(
        "first",
        first,
    )

    builder.add_node(
        "last",
        last,
    )

    builder.add_start_edge("first")
    builder.add_edge("first", "last")
    builder.add_end_edge("last")

    graph = builder.build()

    compiled = graph.compile()

    assert compiled is not None


def test_start_and_end_edges_can_be_chained() -> None:
    """Graph construction should support fluent chaining."""

    builder = RuntimeGraphBuilder()

    async def node(state: dict) -> dict:
        return state

    result = (
        builder
        .add_node("node", node)
        .add_start_edge("node")
        .add_end_edge("node")
    )

    assert result is builder
    assert builder.build().compile() is not None