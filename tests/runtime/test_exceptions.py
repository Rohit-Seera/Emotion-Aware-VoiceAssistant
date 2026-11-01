"""Tests for runtime exceptions."""

import pytest

from app.runtime.exceptions import (
    RuntimeAlreadyStartedError,
    RuntimeCancelledError,
    RuntimeError,
    RuntimeExecutionError,
    RuntimeNotStartedError,
    RuntimeTimeoutError,
)


def test_runtime_error_is_base_exception() -> None:
    """RuntimeError should be the base runtime exception."""

    error = RuntimeError("Runtime failed")

    assert isinstance(error, Exception)
    assert str(error) == "Runtime failed"


def test_runtime_not_started_error_inherits_runtime_error() -> None:
    """RuntimeNotStartedError should inherit RuntimeError."""

    error = RuntimeNotStartedError(
        "Runtime has not been started."
    )

    assert isinstance(error, RuntimeError)
    assert str(error) == "Runtime has not been started."


def test_runtime_already_started_error_inherits_runtime_error() -> None:
    """RuntimeAlreadyStartedError should inherit RuntimeError."""

    error = RuntimeAlreadyStartedError(
        "Runtime has already been started."
    )

    assert isinstance(error, RuntimeError)


def test_runtime_execution_error_inherits_runtime_error() -> None:
    """RuntimeExecutionError should inherit RuntimeError."""

    error = RuntimeExecutionError(
        "Workflow execution failed."
    )

    assert isinstance(error, RuntimeError)


def test_runtime_cancelled_error_inherits_runtime_error() -> None:
    """RuntimeCancelledError should inherit RuntimeError."""

    error = RuntimeCancelledError(
        "Execution was cancelled."
    )

    assert isinstance(error, RuntimeError)


def test_runtime_timeout_error_inherits_runtime_error() -> None:
    """RuntimeTimeoutError should inherit RuntimeError."""

    error = RuntimeTimeoutError(
        "Execution timed out."
    )

    assert isinstance(error, RuntimeError)


def test_runtime_errors_can_be_raised_and_caught() -> None:
    """Runtime-specific errors should be catchable through the base class."""

    with pytest.raises(
        RuntimeError,
        match="Execution timed out",
    ):
        raise RuntimeTimeoutError(
            "Execution timed out."
        )