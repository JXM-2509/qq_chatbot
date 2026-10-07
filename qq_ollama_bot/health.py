"""Read-only, bounded probes; model inference is deliberately never a health probe."""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from typing import Any

import httpx

from qq_ollama_bot.config import Settings

_services_ready = False


def mark_services_ready(ready: bool = True) -> None:
    global _services_ready
    _services_ready = ready


async def probe_ollama(client: httpx.AsyncClient, base_url: str, model: str) -> dict[str, Any]:
    try:
        version = await client.get(f"{base_url}/api/version", timeout=3)
        version.raise_for_status()
        if not version.json().get("version"):
            raise ValueError("invalid version")
    except (httpx.HTTPError, ValueError, AttributeError):
        return {"ollama_reachable": False, "model_available": None}
    try:
        tags = await client.get(f"{base_url}/api/tags", timeout=3)
        tags.raise_for_status()
        names = {item.get("name") for item in tags.json()["models"]}
        available = model in names or (":" not in model and f"{model}:latest" in names)
        return {"ollama_reachable": True, "model_available": available}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {"ollama_reachable": True, "model_available": None}


class HealthMonitor:
    def __init__(self, settings: Settings, get_bots: Callable[[], dict],
                 ready: Callable[[], bool], client: httpx.AsyncClient):
        self.settings, self.get_bots, self.ready, self.client = settings, get_bots, ready, client

    async def snapshot(self) -> dict[str, Any]:
        bots = list(self.get_bots().values())

        async def qq_status() -> dict[str, Any]:
            if not bots:
                return {"onebot_connected": False, "qq_online": None, "qq_probe_ok": False}
            try:
                result = await asyncio.wait_for(bots[0].get_status(), timeout=5)
                if not isinstance(result.get("online"), bool):
                    raise ValueError("invalid online flag")
                return {"onebot_connected": True, "qq_online": result["online"], "qq_probe_ok": True}
            except Exception:
                return {"onebot_connected": True, "qq_online": None, "qq_probe_ok": False}

        qq, model = await asyncio.gather(
            qq_status(), probe_ollama(self.client, self.settings.ollama_base_url, self.settings.ollama_model),
        )
        return {"service": "qq-ollama-bot", "pid": os.getpid(), "ready": self.ready(),
                "checked_at": time.time(), **qq, **model}


def install_health(driver: Any, settings: Settings) -> None:
    from nonebot import get_bots
    client = httpx.AsyncClient(trust_env=False)
    monitor = HealthMonitor(settings, get_bots, lambda: _services_ready, client)

    @driver.server_app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, Any]:
        return await monitor.snapshot()

    @driver.on_shutdown
    async def close_health() -> None:
        mark_services_ready(False)
        await client.aclose()
