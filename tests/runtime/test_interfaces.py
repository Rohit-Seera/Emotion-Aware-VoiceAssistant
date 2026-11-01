"""Tests for runtime interfaces."""

from typing import get_type_hints

import pytest

from app.runtime.interfaces import (
    RuntimeExecutorProtocol,
    RuntimeNodeProtocol,
    RuntimeRegistryProtocol,
    RuntimeStreamProtocol,
    RuntimeWorkflowProtocol,
)


def test_node_protocol_has_execute() -> None:
    """Node protocol should define execute."""

    assert hasattr(
        RuntimeNodeProtocol,
        "execute",
    )


def test_workflow_protocol_has_execute() -> None:
    """Workflow protocol should define execute."""

    assert hasattr(
        RuntimeWorkflowProtocol,
        "execute",
    )


def test_executor_protocol_has_execute() -> None:
    """Executor protocol should define execute."""

    assert hasattr(
        RuntimeExecutorProtocol,
        "execute",
    )


def test_stream_protocol_has_stream() -> None:
    """Stream protocol should define stream."""

    assert hasattr(
        RuntimeStreamProtocol,
        "stream",
    )


def test_registry_protocol_has_register_node() -> None:
    """Registry protocol should define register_node."""

    assert hasattr(
        RuntimeRegistryProtocol,
        "register_node",
    )


def test_registry_protocol_has_get_node() -> None:
    """Registry protocol should define get_node."""

    assert hasattr(
        RuntimeRegistryProtocol,
        "get_node",
    )


def test_registry_protocol_has_has_node() -> None:
    """Registry protocol should define has_node."""

    assert hasattr(
        RuntimeRegistryProtocol,
        "has_node",
    )


def test_node_execute_signature() -> None:
    """Node execute should accept state and context."""

    hints = get_type_hints(
        RuntimeNodeProtocol.execute,
    )

    assert "state" in hints
    assert "context" in hints
    assert "return" in hints


def test_executor_execute_signature() -> None:
    """Executor execute should accept state and context."""

    hints = get_type_hints(
        RuntimeExecutorProtocol.execute,
    )

    assert "state" in hints
    assert "context" in hints
    assert "return" in hints


def test_workflow_execute_signature() -> None:
    """Workflow execute should accept state and context."""

    hints = get_type_hints(
        RuntimeWorkflowProtocol.execute,
    )

    assert "state" in hints
    assert "context" in hints
    assert "return" in hints


def test_stream_signature() -> None:
    """Stream protocol should expose a stream method."""

    hints = get_type_hints(
        RuntimeStreamProtocol.stream,
    )

    assert "state" in hints
    assert "context" in hints
    assert "return" in hints