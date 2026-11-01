"""Exceptions for the memory engine."""

from __future__ import annotations


class MemoryError(Exception):
    """Base exception for memory engine errors."""


class MemoryNotFoundError(MemoryError):
    """Raised when a requested memory does not exist."""

    def __init__(self, memory_id: object) -> None:
        super().__init__(
            f"Memory '{memory_id}' was not found."
        )
        self.memory_id = memory_id


class MemoryConfigurationError(MemoryError):
    """Raised when memory engine configuration is invalid."""


class MemoryProcessingError(MemoryError):
    """Raised when memory processing fails."""