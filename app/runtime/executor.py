"""Runtime execution coordinator."""

from __future__ import annotations

import asyncio
from typing import Protocol

from app.core.settings import settings
from app.runtime.context import RuntimeContext
from app.runtime.exceptions import (
    RuntimeCancelledError,
    RuntimeExecutionError,
    RuntimeTimeoutError,
)
from app.runtime.metrics import RuntimeMetrics
from app.runtime.result import RuntimeResult
from app.runtime.state import RuntimeState
from app.runtime.types import RuntimeTokenUsage


class RuntimeWorkflowProtocol(Protocol):
    """Contract for an executable runtime workflow."""

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeState | RuntimeResult:
        """Execute the workflow."""
        ...


class RuntimeExecutor:
    """Execute runtime workflows and normalize their results."""

    def __init__(
        self,
        workflow: RuntimeWorkflowProtocol,
        *,
        timeout: float | None = None,
        metrics: RuntimeMetrics | None = None,
    ) -> None:
        if timeout is not None and timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        self._workflow = workflow
        self._timeout = (
            float(timeout)
            if timeout is not None
            else float(settings.runtime.timeout)
        )
        self._metrics = metrics

    @property
    def workflow(self) -> RuntimeWorkflowProtocol:
        """Return the configured workflow."""

        return self._workflow

    @property
    def timeout(self) -> float:
        """Return the execution timeout."""

        return self._timeout

    @property
    def metrics(self) -> RuntimeMetrics | None:
        """Return the configured metrics collector."""

        return self._metrics

    async def execute(
        self,
        state: RuntimeState,
        context: RuntimeContext,
    ) -> RuntimeResult:
        """Execute the workflow and return a runtime result."""

        context.state = state

        try:
            result = await asyncio.wait_for(
                self._workflow.execute(
                    state,
                    context,
                ),
                timeout=self._timeout,
            )

        except asyncio.CancelledError as error:
            state.error = "Runtime execution was cancelled."
            self._record_cancellation()

            raise RuntimeCancelledError(
                "Runtime execution was cancelled."
            ) from error

        except TimeoutError as error:
            state.error = (
                "Runtime execution exceeded the configured timeout."
            )
            self._record_failure()

            raise RuntimeTimeoutError(
                "Runtime execution exceeded the configured timeout."
            ) from error

        except Exception as error:
            state.error = str(error)
            self._record_failure()

            raise RuntimeExecutionError(
                f"Workflow execution failed: {error}"
            ) from error

        result = self._normalize_result(
            result,
            state,
        )

        if result.success:
            self._record_success()
        else:
            self._record_failure()

        self._record_tokens(
            result.token_usage,
        )

        return result

    def _normalize_result(
        self,
        result: RuntimeState | RuntimeResult,
        state: RuntimeState,
    ) -> RuntimeResult:
        """Convert workflow output into RuntimeResult."""

        if isinstance(result, RuntimeResult):
            return result

        if isinstance(result, RuntimeState):
            return RuntimeResult(
                request_id=result.request_id,
                workflow=result.workflow,
                status=result.status,
                output=result.output,
                error=result.error,
                metadata=dict(result.metadata),
                token_usage=RuntimeTokenUsage(
                    prompt_tokens=(
                        result.token_usage.prompt_tokens
                    ),
                    completion_tokens=(
                        result.token_usage.completion_tokens
                    ),
                    total_tokens=(
                        result.token_usage.total_tokens
                    ),
                ),
            )

        raise RuntimeExecutionError(
            "Workflow execute() must return "
            "RuntimeState or RuntimeResult."
        )

    def _record_success(self) -> None:
        """Record a successful execution."""

        if self._metrics is not None:
            self._metrics.record_success()

    def _record_failure(self) -> None:
        """Record a failed execution."""

        if self._metrics is not None:
            self._metrics.record_failure()

    def _record_cancellation(self) -> None:
        """Record a cancelled execution."""

        if self._metrics is not None:
            self._metrics.record_cancellation()

    def _record_tokens(
        self,
        usage: RuntimeTokenUsage,
    ) -> None:
        """Record consumed tokens."""

        if self._metrics is not None:
            self._metrics.record_tokens(
                usage.total_tokens,
            )