import asyncio

import pytest

from qq_ollama_bot.services.forward_queue import ForwardQueue


@pytest.mark.asyncio
async def test_fifo_running_key_dedup_and_capacity():
    started, release = asyncio.Event(), asyncio.Event()
    processed = []

    async def process(job):
        processed.append(job)
        if job == 1:
            started.set()
            await release.wait()

    async def fail(job, exc):
        pytest.fail("unexpected queue error")

    queue = ForwardQueue(process, fail, max_pending=2)
    assert queue.enqueue("a", 1) == "closed"
    queue.start()
    try:
        assert queue.enqueue("a", 1) == "queued"
        await asyncio.wait_for(started.wait(), 1)
        assert queue.enqueue("a", 1) == "duplicate"
        assert queue.enqueue("b", 2) == "queued"
        assert queue.enqueue("c", 3) == "queued"
        assert queue.enqueue("d", 4) == "full"
        release.set()
        await asyncio.wait_for(queue.join(), 1)
        assert processed == [1, 2, 3]
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_job_failure_does_not_stop_queue_and_shutdown_discards_pending():
    failures, processed = [], []

    async def process(job):
        if job == "bad":
            raise ValueError("bad")
        processed.append(job)

    async def fail(job, exc):
        failures.append(job)

    queue = ForwardQueue(process, fail, max_pending=2)
    queue.start()
    assert queue.enqueue("a", "bad") == "queued"
    assert queue.enqueue("b", "good") == "queued"
    await asyncio.wait_for(queue.join(), 1)
    assert failures == ["bad"] and processed == ["good"]
    await queue.stop()
    assert queue.enqueue("c", "new") == "closed"
    queue.start()
    queue.enqueue("d", "discarded")
    await queue.stop()
    await asyncio.wait_for(queue.join(), 1)
    queue.start()
    await queue.stop()
    assert processed == ["good"]


@pytest.mark.asyncio
async def test_shutdown_cancels_active_job_and_clears_pending_keys():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def process(job):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def fail(job, exc):
        pytest.fail("Shutdown cancellation must not be reported as a job failure")

    queue = ForwardQueue(process, fail, max_pending=2)
    queue.start()
    queue.enqueue("active", 1)
    await asyncio.wait_for(started.wait(), 1)
    queue.enqueue("pending", 2)
    await queue.stop()
    assert cancelled.is_set()
    await asyncio.wait_for(queue.join(), 1)
    queue.start()
    assert queue.enqueue("pending", 3) == "queued"
    await queue.stop()
