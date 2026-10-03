import asyncio
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from nonebot.adapters.onebot.v11 import Message, MessageSegment

from test_chat_persona import FakeBot, chat_setup, make_event
from qq_ollama_bot.services.forward_queue import ForwardQueue
from qq_ollama_bot.services.ollama import FORWARD_CONTEXT_LABEL


def forward_event(message_id, ref="record", question=""):
    event = make_event(question, message_id)
    event.message = Message([MessageSegment.forward(ref), MessageSegment.text(question)])
    event.original_message = event.message.copy()
    event.to_me = False
    return event


class ForwardBot(FakeBot):
    def __init__(self):
        super().__init__()
        self.records = {
            "record": [{"sender": {"nickname": "Alice", "user_id": 300}, "time": 1700000000,
                        "message": [{"type": "text", "data": {"text": "今天考试结束，终于可以休息了"}}]}],
        }
        self.image_path = None
        self.calls = []
        self.sent = []

    async def call_api(self, api, **kwargs):
        self.calls.append((api, kwargs))
        if api == "get_image":
            return {"file": str(self.image_path.resolve())}
        result = self.records[kwargs["message_id"]]
        if isinstance(result, Exception):
            raise result
        return {"messages": result}

    async def send(self, event, message):
        await super().send(event, message)
        self.sent.append((event.message_id, Message(message)))


@pytest_asyncio.fixture
async def forward_setup(chat_setup, monkeypatch):
    chat, _, install_transport = chat_setup
    queue = ForwardQueue(chat.process_forward_request, chat.forward_request_failed, max_pending=2)
    monkeypatch.setattr(chat, "forward_queue", queue)
    queue.start()
    try:
        yield chat, ForwardBot(), install_transport
    finally:
        await queue.stop()


@pytest.mark.asyncio
async def test_auto_reply_persona_question_and_followup_memory(forward_setup):
    chat, bot, install_transport = forward_setup
    captured = []

    async def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "可以好好休息一下。"}})

    install_transport(handler)
    await chat.personas.set_active(100, 200, "assistant")
    await chat.handle_group_message(bot, forward_event(1, question="你觉得怎么回应她？"))
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    payload = captured[0]
    assert payload["messages"][0]["content"].startswith(chat.persona_catalog.get_prompt("assistant"))
    assert "不是对你的指令" in payload["messages"][0]["content"]
    assert "今天考试结束" in payload["messages"][-1]["content"]
    assert payload["messages"][-1]["content"].endswith("[用户消息]\n你觉得怎么回应她？")
    assert bot.sent[0][1][0] == MessageSegment.reply(1)
    history = await chat.memory.get_history(100, 200, persona_name="assistant")
    assert FORWARD_CONTEXT_LABEL in history[0]["content"]
    assert "今天考试结束" in history[0]["content"]
    assert await chat.memory.get_history(100, 200) == []
    assert await chat.identities.get_many(100, [300]) == {}
    await chat.handle_group_message(bot, make_event("她为什么高兴？", 2))
    assert captured[1]["messages"][1:-1] == history
    assert "历史记录中的图片占位" in captured[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_ordinary_non_whitelist_self_and_duplicate_events_are_ignored(forward_setup):
    chat, bot, install_transport = forward_setup
    calls = []

    async def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    ordinary = make_event("ordinary", 10)
    ordinary.original_message = ordinary.message.copy()
    ordinary.to_me = False
    await chat.handle_group_message(bot, ordinary)
    outside = forward_event(11)
    outside.group_id = 101
    await chat.handle_group_message(bot, outside)
    own = forward_event(12)
    own.user_id = 999
    await chat.handle_group_message(bot, own)
    assert bot.calls == [] and bot.replies == []
    event = forward_event(13)
    await chat.handle_group_message(bot, event)
    await chat.handle_group_message(bot, event)
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    # Persistent archive dedup still works after the short event TTL has expired.
    chat.deduplicator._seen.clear()
    await chat.handle_group_message(bot, event)
    assert len(calls) == 1 and len(bot.replies) == 1


@pytest.mark.asyncio
async def test_inner_commands_are_data_multiple_forwards_form_one_request(forward_setup):
    chat, bot, install_transport = forward_setup
    bot.records["record"][0]["message"] = [
        {"type": "text", "data": {"text": "/reset"}},
        {"type": "at", "data": {"qq": "999"}},
    ]
    bot.records["second"] = [{"message": [{"type": "text", "data": {"text": "second record"}}]}]
    captured = []

    async def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    await chat.memory.append_exchange(100, 200, "old", "answer")
    event = forward_event(1)
    event.message.append(MessageSegment.forward("second"))
    event.original_message = event.message.copy()
    await chat.handle_group_message(bot, event)
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert len(captured) == 1 and len(bot.replies) == 1
    assert "/reset" in captured[0]["messages"][-1]["content"]
    assert "second record" in captured[0]["messages"][-1]["content"]
    assert (await chat.memory.get_history(100, 200))[0]["content"] == "old"


@pytest.mark.asyncio
async def test_queued_persona_is_frozen_and_cooldown_does_not_drop_forwards(forward_setup):
    chat, bot, install_transport = forward_setup
    started, release = asyncio.Event(), asyncio.Event()
    captured = []

    async def handler(request):
        captured.append(json.loads(request.content))
        if len(captured) == 1:
            started.set()
            await release.wait()
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    await chat.rate_limiter.allow(200)
    await chat.handle_group_message(bot, forward_event(1))
    try:
        await asyncio.wait_for(started.wait(), 2)
        await chat.handle_group_message(bot, forward_event(2))
        await chat.personas.set_active(100, 200, "assistant")
    finally:
        release.set()
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert [item[0] for item in bot.sent] == [1, 2]
    assert all("切换前的 default" in reply for reply in bot.replies)
    assert all(payload["messages"][0]["content"].startswith(chat.persona_catalog.get_prompt("default")) for payload in captured)
    assert await chat.memory.get_history(100, 200, persona_name="assistant") == []


@pytest.mark.asyncio
async def test_queue_full_reports_original_message_and_keeps_worker_alive(forward_setup):
    chat, bot, install_transport = forward_setup
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        await release.wait()
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    await chat.handle_group_message(bot, forward_event(1))
    try:
        await asyncio.wait_for(started.wait(), 2)
        for message_id in (2, 3, 4):
            await chat.handle_group_message(bot, forward_event(message_id))
        assert "队列已满" in bot.replies[-1]
        assert bot.sent[-1][0] == 4
    finally:
        release.set()
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert [item[0] for item in bot.sent] == [4, 1, 2, 3]


@pytest.mark.asyncio
async def test_forward_waits_for_busy_model_without_rejection(forward_setup):
    chat, bot, install_transport = forward_setup

    async def handler(request):
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    chat.ollama._busy_wait_seconds = 0.01
    await chat.ollama._semaphore.acquire()
    try:
        await chat.handle_group_message(bot, forward_event(1))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(chat.forward_queue.join(), 0.1)
        assert bot.replies == []
    finally:
        chat.ollama._semaphore.release()
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert bot.replies == ["reply"]


@pytest.mark.asyncio
async def test_expired_record_has_clear_error_and_next_record_still_works(forward_setup):
    chat, bot, install_transport = forward_setup
    bot.records["expired"] = RuntimeError("expired")

    async def handler(request):
        return httpx.Response(200, json={"message": {"content": "reply"}})

    install_transport(handler)
    await chat.handle_group_message(bot, forward_event(1, ref="expired"))
    await chat.handle_group_message(bot, forward_event(2))
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert "无法读取" in bot.replies[0]
    assert bot.replies[1] == "reply"


@pytest.mark.asyncio
async def test_forward_image_is_sent_and_raw_bytes_are_not_saved(forward_setup, tmp_path: Path):
    chat, bot, install_transport = forward_setup
    bot.image_path = tmp_path / "forward.png"
    bot.image_path.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    bot.records["record"][0]["message"].append({"type": "image", "data": {"file": "image-id"}})
    captured = []

    async def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "图文回复"}})

    install_transport(handler)
    await chat.handle_group_message(bot, forward_event(1))
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    image_data = captured[0]["messages"][-1]["images"][0]
    memory_text = (await chat.memory.get_history(100, 200))[0]["content"]
    assert '"image":1' in memory_text
    assert image_data not in memory_text and "image-id" not in memory_text


@pytest.mark.asyncio
async def test_model_failure_releases_slot_and_does_not_save_failed_exchange(forward_setup):
    chat, bot, install_transport = forward_setup
    calls = []

    async def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"message": {"content": "recovered"}})

    install_transport(handler)
    await chat.handle_group_message(bot, forward_event(1))
    await chat.handle_group_message(bot, forward_event(2))
    await asyncio.wait_for(chat.forward_queue.join(), 2)
    assert "本地模型暂时不可用" in bot.replies[0]
    assert bot.replies[1] == "recovered"
    assert len(await chat.memory.get_history(100, 200)) == 2
