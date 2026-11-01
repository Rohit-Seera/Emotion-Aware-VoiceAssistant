"""Tests for the runtime graph compiler."""

from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.runtime.graph.builder import RuntimeGraphBuilder
from app.runtime.graph.compiler import RuntimeGraphCompiler


@pytest.fixture
def compiler() -> RuntimeGraphCompiler:
    """Return a graph compiler."""

    return RuntimeGraphCompiler()


def make_builder() -> RuntimeGraphBuilder:
    """Create a simple valid graph builder."""

    builder = RuntimeGraphBuilder()

    async def node(state: dict) -> dict:
        return state

    (
        builder
        .add_node("node", node)
        .add_start_edge("node")
        .add_end_edge("node")
    )

    return builder


def test_compiler_returns_compiled_graph(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiler should return a compiled LangGraph."""

    result = compiler.compile(
        make_builder(),
    )

    assert isinstance(
        result,
        CompiledStateGraph,
    )


@pytest.mark.asyncio
async def test_compiled_graph_can_invoke(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiled graph should be asynchronously executable."""

    graph = compiler.compile(
        make_builder(),
    )

    result = await graph.ainvoke(
        {
            "input": "hello",
        },
    )

    assert result["input"] == "hello"


def test_compiler_rejects_invalid_builder(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiler should reject non-builder objects."""

    with pytest.raises(
        TypeError,
        match="builder must be a RuntimeGraphBuilder",
    ):
        compiler.compile(
            MagicMock(),
        )


def test_compiler_creates_independent_graphs(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Separate builders should produce separate compiled graphs."""

    first = compiler.compile(
        make_builder(),
    )

    second = compiler.compile(
        make_builder(),
    )

    assert first is not second


def test_compiled_graph_exposes_invoke(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiled graph should expose invoke."""

    graph = compiler.compile(
        make_builder(),
    )

    assert callable(graph.invoke)


def test_compiled_graph_exposes_ainvoke(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiled graph should expose async invoke."""

    graph = compiler.compile(
        make_builder(),
    )

    assert callable(graph.ainvoke)


def test_compiled_graph_exposes_stream(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiled graph should expose stream."""

    graph = compiler.compile(
        make_builder(),
    )

    assert callable(graph.stream)


def test_compiled_graph_exposes_astream(
    compiler: RuntimeGraphCompiler,
) -> None:
    """Compiled graph should expose async stream."""

    graph = compiler.compile(
        make_builder(),
    )

    assert callable(graph.astream)