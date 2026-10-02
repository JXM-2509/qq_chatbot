import asyncio
import importlib
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import nonebot
import pytest
import pytest_asyncio
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from qq_ollama_bot.config import Settings
from qq_ollama_bot.controls import EventDeduplicator, UserRateLimiter
from qq_ollama_bot.services.group_messages import GroupMessageStore
from qq_ollama_bot.services.identity import IdentityStore
from qq_ollama_bot.services.memory import MemoryStore
from qq_ollama_bot.services.ollama import OllamaClient, SYSTEM_PROMPT
from qq_ollama_bot.services.persona import PersonaCatalog, PersonaStore


def make_event(text: str, message_id: int) -> SimpleNamespace:
    message = Message(MessageSegment.text(text))
    return SimpleNamespace(
        group_id=100, user_id=200, self_id="999", message_id=message_id,
        time=int(time.time()), to_me=True, message=message,
        original_message=Message([MessageSegment.at(999), *message]),
        sender=SimpleNamespace(nickname="member", card="", role="member"),
    )


class FakeBot:
    def __init__(self) -> None:
        self.replies: list[str] = []

    async def send(self, event, message) -> None:
        self.replies.append(Message(message).extract_plain_text())


@pytest_asyncio.fixture
async def chat_setup(tmp_path: Path, monkeypatch):
    settings = Settings(
        _env_file=None, onebot_access_token="0123456789abcdef",
        allowed_group_ids="100", database_path=tmp_path / "chat.db",
        user_cooldown_seconds=0,
    )
    nonebot.init(_env_file=None)
    with patch("qq_ollama_bot.config.get_settings", return_value=settings):
        chat = importlib.import_module("qq_ollama_bot.plugins.chat")
    await chat.ollama.close()
    memory = MemoryStore(settings.database_path, max_turns=8, character_budget=12_000, retention_days=30)
    personas = PersonaStore(settings.database_path)
    identities = IdentityStore(settings.database_path)
    archive = GroupMessageStore(settings.database_path)
    for service in (memory, personas, identities, archive):
        await service.initialize()
    for name, value in {
        "settings": settings, "memory": memory, "personas": personas,
        "identities": identities, "group_messages": archive,
        "rate_limiter": UserRateLimiter(0), "deduplicator": EventDeduplicator(),
    }.items():
        monkeypatch.setattr(chat, name, value)
    # Keep the real prompt catalog; HTTP requests and QQ replies are captured locally.
    clients = []

    def install_transport(handler):
        client = OllamaClient(
            base_url="http://127.0.0.1:11434", model="test",
            request_timeout_seconds=10, busy_wait_seconds=1,
            transport=httpx.MockTransport(handler),
        )
        clients.append(client)
        monkeypatch.setattr(chat, "ollama", client)

    async def unexpected_request(request):
        pytest.fail("Tests must explicitly install a local mock HTTP handler")

    install_transport(unexpected_request)
    yield chat, FakeBot(), install_transport
    for client in clients:
        await client.close()


@pytest.mark.asyncio
async def test_switch_changes_payload_history_and_restores_cat_memory(chat_setup, monkeypatch) -> None:
    chat, bot, install_transport = chat_setup
    captured = []
    logs = []
    monkeypatch.setattr(chat, "logger", SimpleNamespace(
        info=logs.append, warning=logs.append, error=logs.append
    ))

    async def handler(request):
        payload = json.loads(request.content)
        captured.append(payload)
        answer = "cat answer" if payload["messages"][0]["content"] == SYSTEM_PROMPT else "assistant answer"
        return httpx.Response(200, json={"message": {"content": answer}})

    install_transport(handler)
    await chat.handle_group_message(bot, make_event("cat question", 1))
    await chat.handle_group_message(bot, make_event("/persona assistant", 2))
    await chat.handle_group_message(bot, make_event("assistant question", 3))
    await chat.handle_group_message(bot, make_event("/persona", 4))
    assert bot.replies[-1] == "当前人格：assistant"
    await chat.handle_group_message(bot, make_event("/persona default", 5))
    await chat.handle_group_message(bot, make_event("continue cat question", 6))
    await chat.handle_group_message(bot, make_event("/persona assistant", 7))
    await chat.handle_group_message(bot, make_event("continue assistant question", 8))

    assert len(captured) == 4  # Switching and querying never invoke the model.
    assert captured[1]["messages"][0]["content"] == chat.persona_catalog.get_prompt("assistant")
    assert len(captured[1]["messages"]) == 2
    assert captured[2]["messages"][1:-1] == [
        {"role": "user", "content": "cat question"},
        {"role": "assistant", "content": "cat answer"},
    ]
    assert captured[3]["messages"][1:-1] == [
        {"role": "user", "content": "assistant question"},
        {"role": "assistant", "content": "assistant answer"},
    ]
    context_logs = [line for line in logs if line.startswith("request_context ")]
    fingerprint = hashlib.sha256(
        chat.persona_catalog.get_prompt("assistant").encode("utf-8")
    ).hexdigest()[:12]
    assert "persona_name=assistant" in context_logs[1]
    assert "history_count=0" in context_logs[1]
    assert f"prompt_fingerprint={fingerprint}" in context_logs[1]
    assert chat.persona_catalog.get_prompt("assistant") not in "\n".join(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("switch_back", [False, True])
async def test_switch_during_inference_marks_reply_and_saves_original_persona(
    chat_setup, switch_back: bool
) -> None:
    chat, bot, install_transport = chat_setup
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        await release.wait()
        return httpx.Response(200, json={"message": {"content": "old cat answer"}})

    install_transport(handler)
    pending = asyncio.create_task(chat.handle_group_message(bot, make_event("cat question", 1)))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        await chat.handle_group_message(bot, make_event("/persona assistant", 2))
        if switch_back:
            await chat.handle_group_message(bot, make_event("/persona default", 3))
    finally:
        release.set()
        await pending

    assert bot.replies[-1] == "【这是切换前的 default 人格请求回复】\nold cat answer"
    assert await chat.memory.get_history(100, 200, persona_name="assistant") == []
    assert (await chat.memory.get_history(100, 200))[-1]["content"] == "old cat answer"


@pytest.mark.asyncio
async def test_reset_commands_apply_to_current_or_all_personas(chat_setup) -> None:
    chat, bot, _ = chat_setup
    for persona in ("default", "assistant"):
        await chat.memory.append_exchange(100, 200, "question", "answer", persona_name=persona)
    await chat.personas.set_active(100, 200, "assistant")
    await chat.handle_group_message(bot, make_event("/reset", 1))
    assert await chat.memory.get_history(100, 200, persona_name="assistant") == []
    assert len(await chat.memory.get_history(100, 200)) == 2
    assert "assistant" in bot.replies[-1]
    await chat.handle_group_message(bot, make_event("/reset all", 2))
    assert await chat.memory.get_history(100, 200) == []
    assert await chat.personas.get_active(100, 200) == "assistant"
    assert len(await chat.group_messages.get_recent_messages(100, 200, 10)) == 2


@pytest.mark.asyncio
async def test_missing_prompt_reports_error_without_model_fallback(chat_setup, tmp_path, monkeypatch) -> None:
    chat, bot, install_transport = chat_setup

    async def handler(request):
        pytest.fail("Missing personas must not fall back to the cat prompt")

    install_transport(handler)
    monkeypatch.setattr(chat, "persona_catalog", PersonaCatalog(tmp_path / "missing"))
    await chat.personas.set_active(100, 200, "assistant")
    await chat.handle_group_message(bot, make_event("/persona", 1))
    assert "assistant" in bot.replies[-1] and "提示词不可用" in bot.replies[-1]
    await chat.handle_group_message(bot, make_event("question", 2))
    assert "提示词不可用" in bot.replies[-1]
    assert await chat.memory.get_history(100, 200) == []
    assert await chat.personas.get_active(100, 200) == "assistant"
