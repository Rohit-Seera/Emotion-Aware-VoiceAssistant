"""Tests for the runtime registry."""

from unittest.mock import MagicMock

import pytest

from app.runtime.exceptions import RuntimeError
from app.runtime.registry import RuntimeRegistry


@pytest.fixture
def registry() -> RuntimeRegistry:
    """Return an empty registry."""

    return RuntimeRegistry()


@pytest.fixture
def node() -> MagicMock:
    """Return a mocked runtime node."""

    return MagicMock()


def test_registry_starts_empty(
    registry: RuntimeRegistry,
) -> None:
    """Registry should start without nodes."""

    assert registry.size == 0


def test_register_node(
    registry: RuntimeRegistry,
    node: MagicMock,
) -> None:
    """Registry should store a node."""

    registry.register_node(
        "llm",
        node,
    )

    assert registry.size == 1
    assert registry.has_node("llm") is True
    assert registry.get_node("llm") is node


def test_register_multiple_nodes(
    registry: RuntimeRegistry,
) -> None:
    """Registry should support multiple nodes."""

    first = MagicMock()
    second = MagicMock()

    registry.register_node("llm", first)
    registry.register_node("memory", second)

    assert registry.size == 2
    assert registry.get_node("llm") is first
    assert registry.get_node("memory") is second


def test_register_empty_name_raises(
    registry: RuntimeRegistry,
    node: MagicMock,
) -> None:
    """Empty node names should be rejected."""

    with pytest.raises(
        ValueError,
        match="Node name cannot be empty",
    ):
        registry.register_node(
            "   ",
            node,
        )


def test_register_duplicate_node_raises(
    registry: RuntimeRegistry,
    node: MagicMock,
) -> None:
    """Duplicate node names should be rejected."""

    registry.register_node(
        "llm",
        node,
    )

    with pytest.raises(
        RuntimeError,
        match="Runtime node 'llm' is already registered",
    ):
        registry.register_node(
            "llm",
            MagicMock(),
        )


def test_get_missing_node_raises(
    registry: RuntimeRegistry,
) -> None:
    """Getting an unknown node should raise."""

    with pytest.raises(
        RuntimeError,
        match="Runtime node 'missing' is not registered",
    ):
        registry.get_node("missing")


def test_has_node_returns_false_for_missing_node(
    registry: RuntimeRegistry,
) -> None:
    """has_node should return False for unknown nodes."""

    assert registry.has_node("missing") is False


def test_unregister_node(
    registry: RuntimeRegistry,
    node: MagicMock,
) -> None:
    """Registry should remove a registered node."""

    registry.register_node(
        "llm",
        node,
    )

    registry.unregister_node("llm")

    assert registry.size == 0
    assert registry.has_node("llm") is False


def test_unregister_missing_node_raises(
    registry: RuntimeRegistry,
) -> None:
    """Removing an unknown node should raise."""

    with pytest.raises(
        RuntimeError,
        match="Runtime node 'missing' is not registered",
    ):
        registry.unregister_node("missing")


def test_clear(
    registry: RuntimeRegistry,
) -> None:
    """clear should remove every registered node."""

    registry.register_node(
        "llm",
        MagicMock(),
    )
    registry.register_node(
        "memory",
        MagicMock(),
    )

    registry.clear()

    assert registry.size == 0
    assert registry.has_node("llm") is False
    assert registry.has_node("memory") is False