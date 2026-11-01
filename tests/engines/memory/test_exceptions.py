"""Tests for memory engine exceptions."""

from app.engines.memory.exceptions import (
    MemoryConfigurationError,
    MemoryError,
    MemoryNotFoundError,
    MemoryProcessingError,
)


def test_memory_error_is_base_exception() -> None:
    """All memory errors should inherit from MemoryError."""

    assert issubclass(
        MemoryNotFoundError,
        MemoryError,
    )
    assert issubclass(
        MemoryConfigurationError,
        MemoryError,
    )
    assert issubclass(
        MemoryProcessingError,
        MemoryError,
    )


def test_memory_not_found_error_contains_id() -> None:
    """Not-found error should expose the memory ID."""

    error = MemoryNotFoundError("abc-123")

    assert error.memory_id == "abc-123"
    assert str(error) == "Memory 'abc-123' was not found."


def test_memory_configuration_error() -> None:
    """Configuration errors should be usable as memory errors."""

    error = MemoryConfigurationError("Invalid configuration")

    assert isinstance(error, MemoryError)
    assert str(error) == "Invalid configuration"


def test_memory_processing_error() -> None:
    """Processing errors should be usable as memory errors."""

    error = MemoryProcessingError("Processing failed")

    assert isinstance(error, MemoryError)
    assert str(error) == "Processing failed"