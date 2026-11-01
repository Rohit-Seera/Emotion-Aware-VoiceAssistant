"""Thread-safe counters for workflow executions and token usage."""

from __future__ import annotations

from threading import Lock


class RuntimeMetrics:
    """Keep lightweight process-local runtime counters."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self._cancelled_executions = 0
        self._total_tokens = 0

    @property
    def executions(self) -> int:
        with self._lock:
            return self._executions

    @property
    def successful_executions(self) -> int:
        with self._lock:
            return self._successful_executions

    @property
    def failed_executions(self) -> int:
        with self._lock:
            return self._failed_executions

    @property
    def cancelled_executions(self) -> int:
        with self._lock:
            return self._cancelled_executions

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self._total_tokens

    def record_success(self) -> None:
        with self._lock:
            self._executions += 1
            self._successful_executions += 1

    def record_failure(self) -> None:
        with self._lock:
            self._executions += 1
            self._failed_executions += 1

    def record_cancellation(self) -> None:
        with self._lock:
            self._executions += 1
            self._cancelled_executions += 1

    def record_tokens(self, count: int) -> None:
        if count < 0:
            raise ValueError("Token count cannot be negative")
        with self._lock:
            self._total_tokens += count

    def reset(self) -> None:
        with self._lock:
            self._executions = 0
            self._successful_executions = 0
            self._failed_executions = 0
            self._cancelled_executions = 0
            self._total_tokens = 0
