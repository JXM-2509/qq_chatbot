from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Awaitable, Callable, Generic, Literal, TypeVar


T = TypeVar("T")
EnqueueResult = Literal["queued", "duplicate", "full", "closed"]


class ForwardQueue(Generic[T]):
    """Bounded FIFO; keys stay reserved while a job is queued or running."""

    def __init__(
        self, processor: Callable[[T], Awaitable[None]],
        on_error: Callable[[T, Exception], Awaitable[None]], *, max_pending: int,
    ) -> None:
        self._processor = processor
        self._on_error = on_error
        self._queue: asyncio.Queue[tuple[str, T]] = asyncio.Queue(maxsize=max_pending)
        self._keys: set[str] = set()
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name="forward-replies")

    def enqueue(self, key: str, job: T) -> EnqueueResult:
        if self._worker is None or self._worker.done():
            return "closed"
        if key in self._keys:
            return "duplicate"
        try:
            self._queue.put_nowait((key, job))
        except asyncio.QueueFull:
            return "full"
        self._keys.add(key)
        return "queued"

    async def join(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._keys.clear()

    async def _run(self) -> None:
        while True:
            key, job = await self._queue.get()
            try:
                await self._processor(job)
            except Exception as exc:
                try:
                    await self._on_error(job, exc)
                except Exception:
                    logging.getLogger(__name__).exception("forward_error_reply_failed")
            finally:
                self._keys.discard(key)
                self._queue.task_done()
