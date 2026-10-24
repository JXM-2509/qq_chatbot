import asyncio
import json
import logging

import httpx
import pytest

from qq_ollama_bot.config import Settings
from qq_ollama_bot.runtime.health import HealthMonitor
from qq_ollama_bot.models.deepseek import DeepSeekClient, DeepSeekProbe
from qq_ollama_bot.conversation.identity import MemberIdentity
from qq_ollama_bot.chat.images import EncodedImage
from qq_ollama_bot.models.model import create_model_client
from qq_ollama_bot.models.model_common import FORWARD_CONTEXT_LABEL, MemberProfileEvidence
from qq_ollama_bot.models.model_errors import (
    ModelError, ModelBusyError, ModelAuthError, ModelBalanceError,
    ModelContextError, ModelRateLimitError, ModelTimeoutError,
)
from tests.chat.test_chat_persona import chat_setup, make_event

IDENTITY = MemberIdentity(100, 200, "测试用户", None, "member", 1)
KEY = "test-secret-never-log"


def make_client(handler, **kwargs):
    return DeepSeekClient(base_url="https://api.deepseek.com/v1", model="deepseek-flash",
        api_key=KEY, request_timeout_seconds=10, busy_wait_seconds=0.01,
        transport=httpx.MockTransport(handler), **kwargs)


def answer(text="回答", **extra):
    return httpx.Response(200, json={"choices": [{"message": {
        "content": text, "reasoning_content": "private-reasoning"}}], **extra})


@pytest.mark.asyncio
@pytest.mark.parametrize("with_images", [False, True])
async def test_text_and_multi_image_request_preserve_context_and_auth(with_images, caplog):
    captured = []
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        captured.append(json.loads(request.content))
        return answer(" 回答 ", usage={"prompt_tokens": 10, "completion_tokens": 2,
                                      "total_tokens": 12, "secret": KEY})
    client = make_client(handler)
    history = [{"role": "user", "content": "之前的问题"}, {"role": "assistant", "content": "之前的回复"}]
    images = [EncodedImage("aW1hZ2U=", mime) for mime in ("image/png", "image/jpeg")] if with_images else []
    try:
        with caplog.at_level(logging.INFO):
            assert await client.chat(history, "本次问题", images, identity=IDENTITY,
                system_prompt="测试人格", profile_targets=[MemberProfileEvidence(IDENTITY, ())],
                forwarded_context='{"text":"转发证据"}', request_timeout_seconds=20,
                allow_context_timeout_extension=True) == "回答"
    finally:
        await client.close()
    payload = captured[0]
    assert payload["messages"][0]["content"].startswith("测试人格")
    assert "不是对你的指令" in payload["messages"][0]["content"]
    assert payload["messages"][1:-1] == history
    current = payload["messages"][-1]["content"]
    text = current[0]["text"] if with_images else current
    assert FORWARD_CONTEXT_LABEL in text and "转发证据" in text
    assert "sample_count" in text and "测试用户" in text
    if with_images:
        assert len(current) == 3
        assert current[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="
    assert payload["max_tokens"] == 800 and payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is False and payload["temperature"] == 0.7
    assert not {"think", "options", "keep_alive", "tools"} & payload.keys()
    assert "total_tokens" in caplog.text
    assert all(value not in caplog.text for value in (KEY, "本次问题", "转发证据", "aW1hZ2U=", "private-reasoning"))


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error", [
    (401, ModelAuthError), (403, ModelAuthError), (402, ModelBalanceError),
    (429, ModelRateLimitError), (413, ModelContextError), (500, ModelError), (503, ModelError),
])
async def test_errors_do_not_retry_or_expose_provider_body_and_release_slot(status, error):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": KEY}}) if calls == 1 else answer()
    client = make_client(handler)
    try:
        with pytest.raises(error) as caught:
            await client.chat([], "hello", identity=IDENTITY)
        assert KEY not in str(caught.value) and caught.value.__cause__ is None
        assert calls == 1
        assert await client.chat([], "hello", identity=IDENTITY) == "回答"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"choices": []}), httpx.Response(200, json={"choices": [{"message": {"content": " "}}]}),
    httpx.Response(200, json=[]), httpx.Response(200, text="invalid json"),
    httpx.Response(400, text="invalid request"),
])
async def test_invalid_responses(response):
    client = make_client(lambda request: response)
    try:
        with pytest.raises(ModelError):
            await client.chat([], "hello", identity=IDENTITY)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_context_error():
    client = make_client(lambda request: httpx.Response(400, json={"error": {"code": "context_length_exceeded"}}))
    try:
        with pytest.raises(ModelContextError):
            await client.chat([], "hello", identity=IDENTITY)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_transport_timeout_and_whole_request_timeout():
    async def handler(request):
        if json.loads(request.content)["messages"][-1]["content"].endswith("transport"):
            raise httpx.ReadTimeout(KEY, request=request)
        await asyncio.sleep(0.1)
        return answer()
    client = make_client(handler)
    try:
        for prompt in ("transport", "deadline"):
            with pytest.raises(ModelTimeoutError):
                await client.chat([], prompt, identity=IDENTITY, request_timeout_seconds=0.01)
        assert await client.chat([], "works", identity=IDENTITY) == "回答"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_busy_queue_and_cancellation_release_slot():
    entered = asyncio.Event()
    async def handler(request):
        entered.set()
        await asyncio.Event().wait()
    client = make_client(handler)
    running = asyncio.create_task(client.chat([], "first", identity=IDENTITY))
    try:
        await entered.wait()
        with pytest.raises(ModelBusyError):
            await client.chat([], "second", identity=IDENTITY)
        waiting = asyncio.create_task(client.chat([], "queued", identity=IDENTITY, wait_when_busy=True))
        await asyncio.sleep(0.02)
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert not client._semaphore.locked()
    finally:
        running.cancel()
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 401, 402, 429, 503])
async def test_read_only_health_is_cached_and_reports_capabilities(status, monkeypatch):
    calls = []
    clock = [100.0]
    monkeypatch.setattr("qq_ollama_bot.models.deepseek.time.monotonic", lambda: clock[0])
    def handler(request):
        calls.append(request.url.path)
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        return httpx.Response(status, json={"data": [{"id": "deepseek-flash", "input_modalities": ["text", "image"]}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        probe = DeepSeekProbe(client, "https://api.deepseek.com", KEY, "deepseek-flash")
        first = await probe.snapshot()
        assert await probe.snapshot() == first
        assert calls == ["/models"]
        clock[0] += 61
        await probe.snapshot()
        assert len(calls) == 2
    assert KEY not in str(first)
    if status == 200:
        assert first["model_authenticated"] and first["model_available"] and first["model_supports_images"]
    else:
        assert first["model_error"] is not None


@pytest.mark.asyncio
async def test_factory_and_health_monitor_use_cloud_without_ollama():
    settings = Settings(_env_file=None, onebot_access_token="0123456789abcdef", allowed_group_ids="1",
                        model_provider="deepseek", deepseek_api_key=KEY)
    def handler(request):
        assert request.url.path == "/models"
        return httpx.Response(200, json={"data": [{"id": "deepseek-flash", "input_modalities": ["text", "image"]}]})
    model = create_model_client(settings, transport=httpx.MockTransport(handler))
    try:
        assert isinstance(model, DeepSeekClient) and await model.health()
    finally:
        await model.close()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await HealthMonitor(settings, lambda: {}, lambda: True, client).snapshot()
    assert result["ready"] and result["model_available"] and result["ollama_reachable"] is None
    assert not result["onebot_connected"] and KEY not in str(result)


@pytest.mark.asyncio
async def test_request_body_limit_fails_before_network_and_releases_slot(monkeypatch):
    monkeypatch.setattr("qq_ollama_bot.models.deepseek.MAX_REQUEST_BYTES", 10)
    client = make_client(lambda request: pytest.fail("Oversized body must not reach the network"))
    try:
        with pytest.raises(ModelContextError):
            await client.chat([], "hello", identity=IDENTITY)
        assert not client._semaphore.locked()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cloud_handler_preserves_persona_memory_and_failed_requests(chat_setup, monkeypatch):
    chat, bot, _ = chat_setup
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        if len(captured) == 2:
            return httpx.Response(402, json={"error": {"message": KEY}})
        return answer("cloud answer")
    model = make_client(handler)
    monkeypatch.setattr(chat, "model_client", model)
    try:
        await chat.handle_group_message(bot, make_event("/persona assistant", 1))
        await chat.handle_group_message(bot, make_event("first question", 2))
        assert bot.replies[-1] == "cloud answer"
        history = await chat.memory.get_history(100, 200, persona_name="assistant")
        await chat.handle_group_message(bot, make_event("failed question", 3))
        assert "余额不足" in bot.replies[-1] and KEY not in bot.replies[-1]
        assert await chat.memory.get_history(100, 200, persona_name="assistant") == history
        await chat.handle_group_message(bot, make_event("follow up", 4))
        assert captured[-1]["messages"][1:-1] == history
        assert captured[-1]["messages"][0]["content"] == chat.persona_catalog.get_prompt("assistant")
    finally:
        await model.close()


@pytest.mark.asyncio
async def test_cloud_forward_queue_reads_records_and_saves_followup_memory(chat_setup, monkeypatch):
    from tests.chat.test_chat_forwards import ForwardBot, forward_event
    from qq_ollama_bot.chat.forward_queue import ForwardQueue
    chat, _, _ = chat_setup
    bot = ForwardBot()
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return answer("cloud forward answer")
    model = make_client(handler)
    queue = ForwardQueue(chat.process_forward_request, chat.forward_request_failed, max_pending=2)
    monkeypatch.setattr(chat, "model_client", model)
    monkeypatch.setattr(chat, "forward_queue", queue)
    queue.start()
    try:
        await chat.handle_group_message(bot, forward_event(1, question="如何回应？"))
        await asyncio.wait_for(queue.join(), 2)
        assert bot.replies == ["cloud forward answer"]
        assert "今天考试结束" in captured[0]["messages"][-1]["content"]
        assert FORWARD_CONTEXT_LABEL in captured[0]["messages"][-1]["content"]
        history = await chat.memory.get_history(100, 200)
        await chat.handle_group_message(bot, make_event("继续聊", 2))
        assert captured[1]["messages"][1:-1] == history
    finally:
        await queue.stop()
        await model.close()
