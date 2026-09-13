import pytest

from qq_ollama_bot.controls import EventDeduplicator, UserRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_is_per_user() -> None:
    now = [100.0]
    limiter = UserRateLimiter(15.0, clock=lambda: now[0])
    assert await limiter.allow(1)
    assert not await limiter.allow(1)
    assert await limiter.allow(2)
    now[0] += 15.0
    assert await limiter.allow(1)


@pytest.mark.asyncio
async def test_event_deduplicator_expires_entries() -> None:
    now = [100.0]
    deduplicator = EventDeduplicator(10.0, clock=lambda: now[0])
    assert not await deduplicator.is_duplicate("event")
    assert await deduplicator.is_duplicate("event")
    now[0] += 11.0
    assert not await deduplicator.is_duplicate("event")

