from __future__ import annotations

import asyncio
import time
from collections.abc import Callable


class UserRateLimiter:
    def __init__(
        self,
        cooldown_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._last_request: dict[int, float] = {}
        self._lock = asyncio.Lock()

    async def allow(self, user_id: int) -> bool:
        now = self._clock()
        async with self._lock:
            previous = self._last_request.get(user_id)
            if previous is not None and now - previous < self._cooldown_seconds:
                return False
            self._last_request[user_id] = now
            return True


class EventDeduplicator:
    def __init__(
        self,
        ttl_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._seen: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_duplicate(self, event_id: str) -> bool:
        now = self._clock()
        async with self._lock:
            cutoff = now - self._ttl_seconds
            self._seen = {
                key: timestamp
                for key, timestamp in self._seen.items()
                if timestamp >= cutoff
            }
            if event_id in self._seen:
                return True
            self._seen[event_id] = now
            return False

