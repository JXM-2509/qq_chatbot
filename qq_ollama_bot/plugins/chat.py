from __future__ import annotations

import asyncio
import contextlib
import time

from nonebot import get_driver, logger, on_message
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.rule import Rule

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.controls import EventDeduplicator, UserRateLimiter
from qq_ollama_bot.message_utils import (
    extract_text,
    get_image_segments,
    has_explicit_at,
    split_reply,
)
from qq_ollama_bot.services.images import ImageInputError, load_images
from qq_ollama_bot.services.identity import IdentityStore, MemberIdentity
from qq_ollama_bot.services.memory import MemoryStore
from qq_ollama_bot.services.ollama import OllamaBusyError, OllamaClient, OllamaError


settings = get_settings()
memory = MemoryStore(
    settings.database_path,
    max_turns=settings.history_turns,
    character_budget=settings.history_character_budget,
    retention_days=settings.retention_days,
)
identities = IdentityStore(
    settings.database_path,
    retention_days=settings.retention_days,
)
ollama = OllamaClient(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    request_timeout_seconds=settings.request_timeout_seconds,
    busy_wait_seconds=settings.busy_wait_seconds,
)
rate_limiter = UserRateLimiter(settings.user_cooldown_seconds)
deduplicator = EventDeduplicator()
driver = get_driver()
cleanup_task: asyncio.Task[None] | None = None


async def eligible_group_mention(event: Event) -> bool:
    return (
        isinstance(event, GroupMessageEvent)
        and event.group_id in settings.allowed_group_id_set
        and event.user_id != int(event.self_id)
        and (
            getattr(event, "to_me", False)
            or has_explicit_at(event.message, event.self_id)
        )
    )


chat_matcher = on_message(
    rule=Rule(eligible_group_mention),
    priority=10,
    block=True,
)


async def periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(24 * 60 * 60)
        try:
            removed_messages = await memory.cleanup_expired()
            removed_identities = await identities.cleanup_expired()
            logger.info(
                "storage_cleanup "
                f"messages_removed={removed_messages} "
                f"identities_removed={removed_identities}"
            )
        except Exception as exc:
            logger.error(f"memory_cleanup_failed error={type(exc).__name__}")


@driver.on_startup
async def initialize_services() -> None:
    global cleanup_task
    await memory.initialize()
    await identities.initialize()
    removed_messages = await memory.cleanup_expired()
    removed_identities = await identities.cleanup_expired()
    cleanup_task = asyncio.create_task(periodic_cleanup(), name="memory-cleanup")
    logger.info(
        "services_started "
        f"expired_messages_removed={removed_messages} "
        f"expired_identities_removed={removed_identities}"
    )


@driver.on_shutdown
async def shutdown_services() -> None:
    if cleanup_task is not None:
        cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cleanup_task
    await ollama.close()


async def send_chunks(bot: Bot, event: GroupMessageEvent, text: str) -> None:
    chunks = split_reply(text, settings.reply_chunk_size)
    for index, chunk in enumerate(chunks):
        if index == 0:
            message = Message(
                [MessageSegment.reply(event.message_id), MessageSegment.text(chunk)]
            )
        else:
            await asyncio.sleep(0.5)
            message = MessageSegment.text(chunk)
        await bot.send(event, message)


async def safe_reply(bot: Bot, event: GroupMessageEvent, text: str) -> None:
    try:
        await send_chunks(bot, event, text)
    except Exception as exc:
        logger.error(
            "reply_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )


async def handle_command(
    bot: Bot,
    event: GroupMessageEvent,
    command: str,
) -> bool:
    normalized = command.casefold()
    if normalized == "/help":
        await safe_reply(
            bot,
            event,
            "可用命令：\n/help 查看帮助\n/reset 清除你在当前群的对话记忆\n/status 检查本地模型状态",
        )
        return True
    if normalized == "/reset":
        try:
            await memory.reset_session(event.group_id, event.user_id)
        except Exception as exc:
            logger.error(
                "memory_reset_failed "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"user_id={event.user_id} error={type(exc).__name__}"
            )
            await safe_reply(bot, event, "记忆服务暂时不可用，请稍后重试。")
        else:
            await safe_reply(bot, event, "已清除你在当前群的对话记忆。")
        return True
    if normalized == "/status":
        healthy = await ollama.health()
        message = "机器人和本地模型均可用。" if healthy else "机器人在线，但本地模型暂时不可用。"
        await safe_reply(bot, event, message)
        return True
    return False


@chat_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    event_id = f"{event.self_id}:{event.message_id}"
    if await deduplicator.is_duplicate(event_id):
        return

    identity = MemberIdentity.from_event(event)
    try:
        identity = await identities.upsert(identity)
    except Exception as exc:
        logger.error(
            "identity_write_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )

    text = extract_text(event.message)
    if await handle_command(bot, event, text):
        return

    image_segments = get_image_segments(event.message)
    if not text and not image_segments:
        await safe_reply(bot, event, "请在 @机器人 后输入问题或附上图片。")
        return
    if len(text) > 4_000:
        await safe_reply(bot, event, "单次问题不能超过 4000 个字符。")
        return
    if not await rate_limiter.allow(event.user_id):
        await safe_reply(bot, event, "请求过于频繁，请 15 秒后再试。")
        return

    started_at = time.monotonic()
    logger.info(
        "request_started "
        f"event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} image_count={len(image_segments)}"
    )
    try:
        images = await load_images(
            bot,
            image_segments,
            max_count=settings.image_max_count,
            max_bytes=settings.image_max_bytes,
            total_max_bytes=settings.image_total_max_bytes,
        )
    except ImageInputError as exc:
        await safe_reply(bot, event, str(exc))
        return

    try:
        history = await memory.get_history(event.group_id, event.user_id)
    except Exception as exc:
        logger.error(
            "memory_read_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "记忆服务暂时不可用，请稍后重试。")
        return

    prompt = text or "请描述并回答图片中的内容。"
    try:
        answer = await ollama.chat(history, prompt, images, identity=identity)
    except OllamaBusyError:
        await safe_reply(bot, event, "当前请求较多，请稍后再试。")
        return
    except OllamaError as exc:
        logger.error(
            "ollama_request_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "本地模型暂时不可用，请稍后重试。")
        return

    await safe_reply(bot, event, answer)
    memory_text = prompt
    if images:
        memory_text = f"{prompt}\n[用户发送了 {len(images)} 张图片]"
    try:
        await memory.append_exchange(
            event.group_id,
            event.user_id,
            memory_text,
            answer,
        )
    except Exception as exc:
        logger.error(
            "memory_write_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "request_completed "
        f"event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} elapsed_ms={elapsed_ms}"
    )
