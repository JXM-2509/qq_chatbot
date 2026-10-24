"""Replay a synthetic merged forward through the configured model; never deliver a QQ message."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
from nonebot.adapters.onebot.v11 import MessageSegment

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.chat.forwards import ForwardReader, prepare_forward
from qq_ollama_bot.conversation.identity import MemberIdentity
from qq_ollama_bot.models.model import create_model_client
from qq_ollama_bot.models.model_errors import ModelError


class LocalRecords:
    def __init__(self, image: Path, long_text: bool):
        self.image = image.resolve()
        self.repeats = 8 if long_text else 1

    async def call_api(self, api, **kwargs):
        if api == "get_image":
            return {"file": str(self.image)}
        assert api == "get_forward_msg"
        return {"messages": [
            {"sender": {"nickname": f"成员 {index % 3}"}, "time": 1700000000 + index,
             "message": [{"type": "text", "data": {"text": f"第 {index + 1} 条：今天的本地功能测试。" * self.repeats}},
                         *([{"type": "image", "data": {"file": "local-test.png"}}] if index < 4 else [])]}
            for index in range(24)
        ]}


class TraceTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.transport = httpx.AsyncHTTPTransport()
        self.attempts = []

    async def handle_async_request(self, request):
        payload = json.loads(request.content)
        print(json.dumps({"context_tokens": payload.get("options", {}).get("num_ctx")}), flush=True)
        response = await self.transport.handle_async_request(request)
        self.attempts.append({"context_tokens": payload.get("options", {}).get("num_ctx"),
                              "http_status": response.status_code})
        print(json.dumps({"http_status": response.status_code}), flush=True)
        return response

    async def aclose(self):
        await self.transport.aclose()


async def check(image: Path, long_text: bool, with_history: bool):
    settings = get_settings()
    bot = LocalRecords(image, long_text)
    reader = ForwardReader(bot, max_depth=settings.forward_max_depth,
                           max_messages=settings.forward_message_limit,
                           character_budget=settings.forward_character_budget,
                           timeout_seconds=settings.forward_timeout_seconds)
    evidence = await reader.read([MessageSegment.forward("local-check")])
    prepared = await prepare_forward(bot, evidence, max_images=settings.image_max_count,
                                     image_max_bytes=settings.image_max_bytes,
                                     image_total_max_bytes=settings.image_total_max_bytes,
                                     timeout_seconds=settings.forward_timeout_seconds)
    transport = TraceTransport()
    client = create_model_client(settings, transport=transport)
    history = ([{"role": "user", "content": "先前聊天中的本地历史测试文字。" * 160},
                {"role": "assistant", "content": "已阅读之前的讨论。"}]
               if with_history else [])
    started = time.monotonic()
    result = {"message_count": reader.message_count, "image_count": len(prepared.images),
              "long_text": long_text,
              "with_history": with_history,
              "qq_message_sent": False}
    print(json.dumps({"status": "started", "provider": settings.model_provider, **result}), flush=True)
    try:
        answer = await client.chat(history, "请简短描述这份聊天记录。", prepared.images,
                                   identity=MemberIdentity(0, 0, "本地验证", None, "member", 1),
                                   system_prompt=(Path(__file__).resolve().parents[2] / "qq_ollama_bot/prompts/assistant.md").read_text(encoding="utf-8"),
                                   forwarded_context=prepared.context, wait_when_busy=True,
                                   request_timeout_seconds=settings.forward_model_timeout_seconds,
                                   allow_context_timeout_extension=True)
        result.update(status="success", reply_characters=len(answer))
    except ModelError as exc:
        result.update(status="failed", error=type(exc).__name__)
    finally:
        await client.close()
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    result["attempts"] = transport.attempts
    print(json.dumps(result), flush=True)
    return result["status"] == "success"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--long-text", action="store_true")
    parser.add_argument("--with-history", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(check(args.image, args.long_text, args.with_history)) else 1)
