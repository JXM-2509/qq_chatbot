import asyncio

import httpx
import pytest

from qq_ollama_bot.config import Settings
from qq_ollama_bot.runtime.health import HealthMonitor, probe_ollama


def settings():
    return Settings(_env_file=None, onebot_access_token="0123456789abcdef", allowed_group_ids="1")


def model_transport(model=True):
    def handle(request):
        assert request.url.path in {"/api/version", "/api/tags"}  # Never run an inference probe.
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "test"})
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}] if model else []})
    return httpx.MockTransport(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("online", [True, False])
async def test_health_reports_real_qq_state(online):
    class Bot:
        async def get_status(self):
            return {"online": online, "good": True}
    async with httpx.AsyncClient(transport=model_transport()) as client:
        monitor = HealthMonitor(settings(), lambda: {"1": Bot()}, lambda: True, client)
        result = await monitor.snapshot()
    assert result["ready"] and result["onebot_connected"]
    assert result["qq_online"] is online and result["qq_probe_ok"]
    assert result["model_available"]
    assert "0123456789abcdef" not in str(result)


@pytest.mark.asyncio
async def test_no_bot_and_missing_model_are_distinct():
    async with httpx.AsyncClient(transport=model_transport(False)) as client:
        result = await HealthMonitor(settings(), lambda: {}, lambda: False, client).snapshot()
    assert result["qq_online"] is None
    assert not result["onebot_connected"] and not result["ready"]
    assert result["ollama_reachable"] and result["model_available"] is False


@pytest.mark.asyncio
async def test_api_failure_does_not_claim_qq_is_offline():
    class Bot:
        async def get_status(self):
            raise asyncio.TimeoutError
    async with httpx.AsyncClient(transport=model_transport()) as client:
        result = await HealthMonitor(settings(), lambda: {"1": Bot()}, lambda: True, client).snapshot()
    assert result["qq_online"] is None and not result["qq_probe_ok"]
    assert result["onebot_connected"]


@pytest.mark.asyncio
async def test_ollama_unavailable_and_invalid_tags():
    def unavailable(request):
        raise httpx.ConnectError("offline")
    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as client:
        assert await probe_ollama(client, "http://127.0.0.1:11434", "model") == {
            "ollama_reachable": False, "model_available": None}
    def malformed(request):
        return httpx.Response(200, json={"version": "test"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(malformed)) as client:
        assert await probe_ollama(client, "http://127.0.0.1:11434", "model") == {
            "ollama_reachable": True, "model_available": None}
