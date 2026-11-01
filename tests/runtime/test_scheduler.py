"""Tests for the runtime scheduler."""

import asyncio

import pytest

from app.runtime.scheduler import RuntimeScheduler


@pytest.mark.asyncio
async def test_scheduler_defaults() -> None:
    """Scheduler should use the expected default limits."""

    scheduler = RuntimeScheduler()

    assert scheduler.max_parallel_tasks == 10
    assert scheduler.max_queue_size == 100
    assert scheduler.queue_size == 0


def test_scheduler_rejects_invalid_parallel_limit() -> None:
    """Parallel task limit must be positive."""

    with pytest.raises(
        ValueError,
        match="max_parallel_tasks must be greater than zero",
    ):
        RuntimeScheduler(max_parallel_tasks=0)


def test_scheduler_rejects_invalid_queue_limit() -> None:
    """Queue size must be positive."""

    with pytest.raises(
        ValueError,
        match="max_queue_size must be greater than zero",
    ):
        RuntimeScheduler(max_queue_size=0)


@pytest.mark.asyncio
async def test_run_executes_task() -> None:
    """run should execute and return the task result."""

    scheduler = RuntimeScheduler()

    async def task() -> str:
        return "done"

    result = await scheduler.run(task)

    assert result == "done"


@pytest.mark.asyncio
async def test_run_propagates_task_exception() -> None:
    """run should propagate task exceptions."""

    scheduler = RuntimeScheduler()

    async def task() -> None:
        raise ValueError("task failed")

    with pytest.raises(
        ValueError,
        match="task failed",
    ):
        await scheduler.run(task)


@pytest.mark.asyncio
async def test_submit_executes_task() -> None:
    """submit should schedule the task."""

    scheduler = RuntimeScheduler()

    async def task() -> str:
        return "done"

    submitted = await scheduler.submit(task)
    result = await submitted

    assert result == "done"
    assert scheduler.queue_size == 0


@pytest.mark.asyncio
async def test_submit_multiple_tasks() -> None:
    """Scheduler should support multiple submitted tasks."""

    scheduler = RuntimeScheduler()

    async def task() -> str:
        await asyncio.sleep(0)
        return "done"

    first = await scheduler.submit(task)
    second = await scheduler.submit(task)

    assert await first == "done"
    assert await second == "done"
    assert scheduler.queue_size == 0


@pytest.mark.asyncio
async def test_scheduler_respects_parallel_limit() -> None:
    """Scheduler should limit concurrent execution."""

    scheduler = RuntimeScheduler(
        max_parallel_tasks=1,
    )

    active = 0
    maximum_active = 0

    async def task() -> None:
        nonlocal active
        nonlocal maximum_active

        active += 1
        maximum_active = max(
            maximum_active,
            active,
        )

        await asyncio.sleep(0.01)

        active -= 1

    await asyncio.gather(
        scheduler.run(task),
        scheduler.run(task),
        scheduler.run(task),
    )

    assert maximum_active == 1


@pytest.mark.asyncio
async def test_submit_rejects_full_queue() -> None:
    """Submitting to a full queue should fail."""

    scheduler = RuntimeScheduler(
        max_parallel_tasks=1,
        max_queue_size=1,
    )

    async def task() -> None:
        await asyncio.sleep(0.1)

    first = await scheduler.submit(task)

    try:
        with pytest.raises(
            RuntimeError,
            match="Runtime scheduler queue is full",
        ):
            await scheduler.submit(task)
    finally:
        await first


@pytest.mark.asyncio
async def test_queue_size_returns_zero_after_execution() -> None:
    """Completed submitted tasks should leave the queue empty."""

    scheduler = RuntimeScheduler()

    async def task() -> int:
        return 42

    submitted = await scheduler.submit(task)

    assert await submitted == 42
    assert scheduler.queue_size == 0