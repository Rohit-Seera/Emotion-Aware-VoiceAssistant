"""Runtime-specific exceptions."""

from __future__ import annotations


class RuntimeError(Exception):
    """Base exception for runtime failures."""


class RuntimeNotStartedError(RuntimeError):
    """Raised when an operation requires a started runtime."""


class RuntimeAlreadyStartedError(RuntimeError):
    """Raised when a runtime is started more than once."""


class RuntimeExecutionError(RuntimeError):
    """Raised when workflow execution fails."""


class RuntimeCancelledError(RuntimeError):
    """Raised when runtime execution is cancelled."""


class RuntimeTimeoutError(RuntimeError):
    """Raised when runtime execution exceeds its timeout."""