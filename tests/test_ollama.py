import json

import httpx
import pytest

from qq_ollama_bot.services.images import EncodedImage
from qq_ollama_bot.services.ollama import OllamaClient


@pytest.mark.asyncio
async def test_chat_builds_non_streaming_request_without_tools() -> None:
    captured: dict = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "回答"}})

    client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen3.5:9b",
        request_timeout_seconds=120,
        busy_wait_seconds=1,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.chat(
            [{"role": "user", "content": "上一个问题"}],
            "当前问题",
            [EncodedImage(data="aW1hZ2U=", mime_type="image/png")],
        )
    finally:
        await client.close()

    assert result == "回答"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert "tools" not in captured
    assert captured["messages"][-1]["images"] == ["aW1hZ2U="]

