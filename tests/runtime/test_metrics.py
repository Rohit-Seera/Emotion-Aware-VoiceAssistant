"""Tests for runtime metrics."""

import pytest

from app.runtime.metrics import RuntimeMetrics


def test_metrics_start_at_zero() -> None:
    """Metrics should start at zero."""

    metrics = RuntimeMetrics()

    assert metrics.executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.cancelled_executions == 0
    assert metrics.total_tokens == 0


def test_record_success() -> None:
    """record_success should update execution counters."""

    metrics = RuntimeMetrics()

    metrics.record_success()

    assert metrics.executions == 1
    assert metrics.successful_executions == 1
    assert metrics.failed_executions == 0
    assert metrics.cancelled_executions == 0


def test_record_failure() -> None:
    """record_failure should update execution counters."""

    metrics = RuntimeMetrics()

    metrics.record_failure()

    assert metrics.executions == 1
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 1
    assert metrics.cancelled_executions == 0


def test_record_cancellation() -> None:
    """record_cancellation should update execution counters."""

    metrics = RuntimeMetrics()

    metrics.record_cancellation()

    assert metrics.executions == 1
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.cancelled_executions == 1


def test_metrics_accumulate_multiple_executions() -> None:
    """Metrics should accumulate execution counts."""

    metrics = RuntimeMetrics()

    metrics.record_success()
    metrics.record_success()
    metrics.record_failure()
    metrics.record_cancellation()

    assert metrics.executions == 4
    assert metrics.successful_executions == 2
    assert metrics.failed_executions == 1
    assert metrics.cancelled_executions == 1


def test_record_tokens() -> None:
    """record_tokens should accumulate token usage."""

    metrics = RuntimeMetrics()

    metrics.record_tokens(100)
    metrics.record_tokens(50)

    assert metrics.total_tokens == 150


def test_record_tokens_rejects_negative_values() -> None:
    """Negative token counts should be rejected."""

    metrics = RuntimeMetrics()

    with pytest.raises(
        ValueError,
        match="Token count cannot be negative",
    ):
        metrics.record_tokens(-1)

    assert metrics.total_tokens == 0


def test_reset_clears_metrics() -> None:
    """reset should clear all metrics."""

    metrics = RuntimeMetrics()

    metrics.record_success()
    metrics.record_failure()
    metrics.record_cancellation()
    metrics.record_tokens(150)

    metrics.reset()

    assert metrics.executions == 0
    assert metrics.successful_executions == 0
    assert metrics.failed_executions == 0
    assert metrics.cancelled_executions == 0
    assert metrics.total_tokens == 0