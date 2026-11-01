"""Tests for the runtime graph checkpointer."""

from langgraph.checkpoint.memory import InMemorySaver

from app.runtime.graph.checkpointer import RuntimeCheckpointer


def test_checkpointer_creates_default_saver() -> None:
    """RuntimeCheckpointer should create an InMemorySaver by default."""

    manager = RuntimeCheckpointer()

    assert isinstance(
        manager.checkpointer,
        InMemorySaver,
    )


def test_checkpointer_stores_provided_saver() -> None:
    """RuntimeCheckpointer should preserve a provided saver."""

    saver = InMemorySaver()

    manager = RuntimeCheckpointer(
        checkpointer=saver,
    )

    assert manager.checkpointer is saver


def test_multiple_checkpointers_are_independent() -> None:
    """Default checkpointers should be separate instances."""

    first = RuntimeCheckpointer()
    second = RuntimeCheckpointer()

    assert first.checkpointer is not second.checkpointer


def test_checkpointer_property_returns_same_instance() -> None:
    """The property should return the configured saver."""

    manager = RuntimeCheckpointer()

    first = manager.checkpointer
    second = manager.checkpointer

    assert first is second