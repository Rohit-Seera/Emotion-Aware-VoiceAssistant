"""Runtime task scheduler."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class RuntimeScheduler:
    """Schedule and execute runtime tasks with bounded concurrency."""

    def __init__(
        self,
        *,
        max_parallel_tasks: int = 10,
        max_queue_size: int = 100,
    ) -> None:
        if max_parallel_tasks <= 0:
            raise ValueError(
                "max_parallel_tasks must be greater than zero."
            )

        if max_queue_size <= 0:
            raise ValueError(
                "max_queue_size must be greater than zero."
            )

        self._semaphore = asyncio.Semaphore(
            max_parallel_tasks,
        )
        self._queue: asyncio.Queue[Any] = asyncio.Queue(
            maxsize=max_queue_size,
        )
        self._max_parallel_tasks = max_parallel_tasks
        self._max_queue_size = max_queue_size

    @property
    def max_parallel_tasks(self) -> int:
        """Return the concurrency limit."""

        return self._max_parallel_tasks

    @property
    def max_queue_size(self) -> int:
        """Return the queue capacity."""

        return self._max_queue_size

    @property
    def queue_size(self) -> int:
        """Return the number of queued tasks."""

        return self._queue.qsize()

    async def run(
        self,
        task: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run a task while respecting the concurrency limit."""

        async with self._semaphore:
            return await task()

    async def submit(
        self,
        task: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task[Any]:
        """Submit a task for asynchronous execution."""

        if self._queue.full():
            raise RuntimeError(
                "Runtime scheduler queue is full."
            )

        await self._queue.put(task)

        async def execute() -> Any:
            queued_task = await self._queue.get()

            try:
                return await self.run(queued_task)
            finally:
                self._queue.task_done()

        return asyncio.create_task(execute())