"""Read-only image diagnosis through the local NapCat WebUI debug API.

Never prints credentials, signed URLs, QQ IDs, names or conversation text.
Run from the project directory. No group messages are sent.
"""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.services.forwards import ForwardReader, prepare_forward
from qq_ollama_bot.services.images import ImageInputError, load_images
from qq_ollama_bot.services.identity import MemberIdentity
from qq_ollama_bot.services.ollama import OllamaClient


def emit(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


class DebugBot:
    def __init__(self, client):
        self.client = client
        self.image_metadata = []

    async def call_api(self, action, **params):
        response = await self.client.post("/api/Debug/call", json={
            "adapterName": "debug-primary", "action": action, "params": params,
        })
        response.raise_for_status()
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError("debug_api_failed")
        result = body.get("data")
        if isinstance(result, dict) and "retcode" in result:
            if result.get("retcode") != 0:
                raise RuntimeError("onebot_api_failed")
            result = result.get("data")
        if action == "get_image" and isinstance(result, dict):
            cache = result.get("file")
            self.image_metadata.append({
                "cache_returned": bool(cache),
                "cache_exists": Path(str(cache)).is_file() if cache else False,
                "base64_returned": bool(result.get("base64")),
                "url_returned": bool(result.get("url")),
            })
        return result


async def main(expected_images: int | None = None, test_model: bool = False):
    settings = get_settings()
    config_path = Path(__file__).resolve().parents[2] / "napcat/config/webui.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    token = config.get("token")
    if not isinstance(token, str) or not token:
        emit(stage="auth", result="local_webui_token_unavailable")
        return
    port = int(config.get("port", 6099))
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False,
    ) as client:
        response = await client.post("/api/auth/login", json={
            "hash": hashlib.sha256((token + ".napcat").encode()).hexdigest(),
        })
        response.raise_for_status()
        body = response.json()
        credential = (body.get("data") or {}).get("Credential")
        if not credential:
            emit(stage="auth", result="authentication_failed")
            return
        client.headers["Authorization"] = f"Bearer {credential}"
        emit(stage="auth", result="ok")
        bot = DebugBot(client)
        group_id = next(iter(settings.allowed_group_id_set))
        history = await bot.call_api(
            "get_group_msg_history", group_id=str(group_id), count=10, parse_mult_msg=True,
        )
        emit(stage="history", result_fields=sorted(history) if isinstance(history, dict) else [],
             message_count=len((history or {}).get("messages", [])))
        candidates = []
        for record in (history or {}).get("messages", []):
            message = record.get("message")
            if not isinstance(message, list):
                continue
            forwards = [item for item in message if isinstance(item, dict) and item.get("type") == "forward"]
            if forwards:
                candidates.append((int(record.get("time", 0)), forwards))
        if not candidates:
            emit(stage="history", result="no_recent_forward")
            return
        from nonebot.adapters.onebot.v11 import MessageSegment

        for _, forwards in sorted(candidates, key=lambda item: item[0], reverse=True):
            emit(stage="forward_fetch_started", outer_forward_count=len(forwards))
            reader = ForwardReader(
                bot, max_depth=settings.forward_max_depth, max_messages=settings.forward_message_limit,
                character_budget=settings.forward_character_budget, timeout_seconds=settings.forward_timeout_seconds,
            )
            evidence = await reader.read([MessageSegment("forward", item.get("data", {})) for item in forwards])
            images = [part.image for entry in evidence.entries for part in entry.parts if part.image is not None]
            if expected_images is None or len(images) == expected_images:
                break
        else:
            emit(stage="history", result="no_forward_with_requested_image_count")
            return
        emit(stage="forward", image_count=len(images), parsed_entries=len(evidence.entries))
        if test_model:
            prepared = await prepare_forward(
                bot, evidence, max_images=settings.image_max_count,
                image_max_bytes=settings.image_max_bytes,
                image_total_max_bytes=settings.image_total_max_bytes,
                timeout_seconds=settings.forward_timeout_seconds,
            )
            model = OllamaClient(
                base_url=settings.ollama_base_url, model=settings.ollama_model,
                request_timeout_seconds=settings.request_timeout_seconds,
                busy_wait_seconds=settings.busy_wait_seconds,
            )
            emit(stage="model_started", image_count=len(prepared.images),
                 timeout_seconds=settings.forward_model_timeout_seconds)
            started = time.monotonic()
            try:
                answer = await model.chat(
                    [], "请阅读这份合并转发聊天记录，沿用当前人格自然简短回应其中的内容。",
                    prepared.images, identity=MemberIdentity(0, 0, "测试发送者", "", "member", 0),
                    forwarded_context=prepared.context, wait_when_busy=True,
                    request_timeout_seconds=settings.forward_model_timeout_seconds,
                )
                emit(stage="model_result", result="ok", reply_characters=len(answer),
                     elapsed_seconds=round(time.monotonic() - started, 1))
            finally:
                await model.close()
            return
        for index, segment in enumerate(images[:settings.image_max_count], 1):
            data = segment.data
            url = data.get("url")
            host = urlsplit(url).hostname if isinstance(url, str) else None
            emit(stage="image_metadata", image_index=index, fields=sorted(data), url_host=host,
                 filename_suffix=Path(str(data.get("file", ""))).suffix[:12])
            try:
                loaded = await load_images(
                    bot, [segment], max_count=1, max_bytes=settings.image_max_bytes,
                    total_max_bytes=settings.image_total_max_bytes,
                )
                emit(stage="image_result", image_index=index, result="ok", mime_type=loaded[0].mime_type)
            except ImageInputError as exc:
                emit(stage="image_result", image_index=index, result="failed", reason=exc.code)
            if bot.image_metadata:
                emit(stage="cache_metadata", image_index=index, **bot.image_metadata[-1])


async def test_local_images(paths: list[str]):
    """Exercise the same model timeout using user-supplied local screenshots."""
    from nonebot.adapters.onebot.v11 import MessageSegment

    class LocalImages:
        async def call_api(self, action, **params):
            return {"file": str(Path(params["file"]).resolve())}

    settings = get_settings()
    images = await load_images(
        LocalImages(), [MessageSegment("image", {"file": path}) for path in paths],
        max_count=settings.image_max_count, max_bytes=settings.image_max_bytes,
        total_max_bytes=settings.image_total_max_bytes,
    )
    model = OllamaClient(
        base_url=settings.ollama_base_url, model=settings.ollama_model,
        request_timeout_seconds=settings.request_timeout_seconds,
        busy_wait_seconds=settings.busy_wait_seconds,
    )
    emit(stage="model_started", image_count=len(images), source="user_screenshots",
         timeout_seconds=settings.forward_model_timeout_seconds)
    started = time.monotonic()
    try:
        answer = await model.chat(
            [], "简短说明截图中的机器人遇到了什么问题。", images,
            identity=MemberIdentity(0, 0, "测试发送者", "", "member", 0),
            forwarded_context=json.dumps({"messages": [{"sender": "测试发送者", "content": [
                {"image": index, "status": "已附图，可识别"} for index in range(1, len(images) + 1)
            ]}]}, ensure_ascii=False),
            request_timeout_seconds=settings.forward_model_timeout_seconds,
        )
        emit(stage="model_result", result="ok", reply_characters=len(answer),
             elapsed_seconds=round(time.monotonic() - started, 1))
    finally:
        await model.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-count", type=int, help="Select the newest forward with this many images")
    parser.add_argument("--model", action="store_true", help="Also test local model inference; never send a group reply")
    parser.add_argument("--local-images", nargs="+", help="Test inference using supplied local screenshots")
    options = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        if options.local_images:
            asyncio.run(test_local_images(options.local_images))
        else:
            asyncio.run(main(options.image_count, options.model))
    except Exception as error:
        emit(stage="diagnosis", result="failed", error_type=type(error).__name__)
