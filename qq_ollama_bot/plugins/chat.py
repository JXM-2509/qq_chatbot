from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

from nonebot import get_driver, logger, on_message
from nonebot.adapters import Event
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.rule import Rule

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.controls import EventDeduplicator, UserRateLimiter
from qq_ollama_bot.message_utils import (
    extract_text,
    get_at_user_ids,
    get_image_segments,
    should_reply_to_group_message,
    split_reply,
)
from qq_ollama_bot.services.group_messages import (
    GroupMessageStore,
    trim_recent_messages,
)
from qq_ollama_bot.services.images import ImageInputError, load_images
from qq_ollama_bot.services.identity import IdentityStore, MemberIdentity
from qq_ollama_bot.services.memory import MemoryStore
from qq_ollama_bot.services.persona import PersonaCatalog, PersonaStore
from qq_ollama_bot.services.ollama import (
    MemberProfileEvidence,
    OllamaBusyError,
    OllamaClient,
    OllamaError,
)


settings = get_settings()
memory = MemoryStore(
    settings.database_path,
    max_turns=settings.history_turns,
    character_budget=settings.history_character_budget,
    retention_days=settings.retention_days,
)
identities = IdentityStore(settings.database_path)
group_messages = GroupMessageStore(settings.database_path)
personas = PersonaStore(settings.database_path)
persona_catalog = PersonaCatalog(Path(__file__).resolve().parents[1] / "prompts")
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


async def eligible_group_message(event: Event) -> bool:
    return (
        isinstance(event, GroupMessageEvent)
        and event.group_id in settings.allowed_group_id_set
        and event.user_id != int(event.self_id)
    )


chat_matcher = on_message(
    rule=Rule(eligible_group_message),
    priority=10,
    block=True,
)


async def periodic_cleanup() -> None:
    while True:
        await asyncio.sleep(24 * 60 * 60)
        try:
            removed_messages = await memory.cleanup_expired()
            logger.info(f"memory_cleanup messages_removed={removed_messages}")
        except Exception as exc:
            logger.error(f"memory_cleanup_failed error={type(exc).__name__}")


@driver.on_startup
async def initialize_services() -> None:
    global cleanup_task
    await memory.initialize()
    await identities.initialize()
    await group_messages.initialize()
    await personas.initialize()
    removed_messages = await memory.cleanup_expired()
    cleanup_task = asyncio.create_task(periodic_cleanup(), name="memory-cleanup")
    logger.info(
        "services_started "
        f"expired_messages_removed={removed_messages}"
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
    normalized = command.strip().casefold()
    if normalized == "/help":
        await safe_reply(
            bot,
            event,
            "可用命令：\n"
            "/help 查看帮助\n"
            "/reset 清除你在当前群的对话记忆\n"
            "/status 检查本地模型状态\n"
            "/persona 查看当前人格；/persona list 查看可用人格\n"
            "/persona <名称> 切换你在当前群的人格\n"
            "@机器人 @群友 问题：基于该群友的近期发言回答",
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
    if normalized == "/persona" or normalized.startswith("/persona "):
        parts = normalized.split()
        try:
            if len(parts) == 1:
                name = await personas.get_active(event.group_id, event.user_id)
                if persona_catalog.get_prompt(name) is None:
                    name = "default"
                reply = f"当前人格：{name}"
            elif len(parts) == 2 and parts[1] == "list":
                reply = "可用人格：" + "、".join(persona_catalog.names())
            elif len(parts) == 2:
                name = parts[1]
                if persona_catalog.get_prompt(name) is None:
                    reply = "没有这个人格。发送 /persona list 查看可用人格。"
                else:
                    await personas.set_active(event.group_id, event.user_id, name)
                    reply = f"已切换为 {name}，仅影响你在当前群的对话。"
            else:
                reply = "用法：/persona、/persona list、/persona <名称>"
        except Exception as exc:
            logger.error(
                "persona_command_failed "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"user_id={event.user_id} error={type(exc).__name__}"
            )
            reply = "人格设置暂时不可用，请稍后重试。"
        await safe_reply(bot, event, reply)
        return True
    return False


async def load_profile_targets(
    bot: Bot,
    event: GroupMessageEvent,
    target_user_ids: list[int],
) -> tuple[MemberProfileEvidence, ...]:
    try:
        target_identities = await identities.get_many(event.group_id, target_user_ids)
    except Exception as exc:
        logger.error(
            "identity_read_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"error={type(exc).__name__}"
        )
        target_identities = {}

    for target_user_id in target_user_ids:
        if target_user_id in target_identities:
            continue
        try:
            member_info = await bot.get_group_member_info(
                group_id=event.group_id,
                user_id=target_user_id,
                no_cache=True,
            )
            identity = MemberIdentity.from_group_member_info(
                event.group_id,
                target_user_id,
                member_info,
            )
            target_identities[target_user_id] = identity
            try:
                target_identities[target_user_id] = await identities.upsert(identity)
            except Exception as exc:
                logger.error(
                    "identity_write_failed "
                    f"event_id={event.message_id} group_id={event.group_id} "
                    f"user_id={target_user_id} error={type(exc).__name__}"
                )
        except Exception as exc:
            logger.warning(
                "group_member_info_failed "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"target_user_id={target_user_id} error={type(exc).__name__}"
            )

    per_target_budget = max(
        1,
        settings.profile_character_budget // len(target_user_ids),
    )
    profiles: list[MemberProfileEvidence] = []
    for target_user_id in target_user_ids:
        identity = target_identities.get(target_user_id) or MemberIdentity(
            group_id=event.group_id,
            user_id=target_user_id,
            nickname=None,
            card=None,
            role=None,
            updated_at=int(time.time()),
        )
        messages = await group_messages.get_recent_messages(
            event.group_id,
            target_user_id,
            settings.profile_message_limit,
        )
        profiles.append(
            MemberProfileEvidence(
                identity=identity,
                messages=trim_recent_messages(messages, per_target_budget),
            )
        )
    return tuple(profiles)


@chat_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    event_id = f"{event.self_id}:{event.group_id}:{event.message_id}"
    if await deduplicator.is_duplicate(event_id):
        return

    identity = MemberIdentity.from_event(event, observed_at=event.time)
    try:
        identity = await identities.upsert(identity)
    except Exception as exc:
        logger.error(
            "identity_write_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )

    original_message = event.original_message
    original_text = extract_text(original_message)
    mentioned_user_ids = get_at_user_ids(original_message)
    try:
        await group_messages.append_event(
            group_id=event.group_id,
            message_id=event.message_id,
            user_id=event.user_id,
            content=original_text,
            created_at=event.time,
            mentioned_user_ids=mentioned_user_ids,
        )
    except Exception as exc:
        logger.error(
            "group_message_archive_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )

    should_reply = should_reply_to_group_message(
        original_message,
        event.self_id,
        to_me=bool(getattr(event, "to_me", False)),
    )
    if not should_reply:
        return

    text = extract_text(event.message)
    if await handle_command(bot, event, text):
        return

    target_user_ids = get_at_user_ids(
        original_message,
        exclude={int(event.self_id)},
    )
    if len(target_user_ids) > settings.profile_target_limit:
        await safe_reply(
            bot,
            event,
            f"一次最多查询 {settings.profile_target_limit} 位群友，请减少 @ 的人数。",
        )
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

    profile_targets: tuple[MemberProfileEvidence, ...] = ()
    if target_user_ids:
        try:
            profile_targets = await load_profile_targets(bot, event, target_user_ids)
        except Exception as exc:
            logger.error(
                "profile_evidence_read_failed "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"error={type(exc).__name__}"
            )
            await safe_reply(bot, event, "群聊记录暂时无法读取，请稍后重试。")
            return

    prompt = text or "请描述并回答图片中的内容。"
    try:
        persona_name = await personas.get_active(event.group_id, event.user_id)
        system_prompt = persona_catalog.get_prompt(persona_name)
        if system_prompt is None:
            logger.warning(f"persona_unavailable name={persona_name}")
            system_prompt = persona_catalog.get_prompt("default")
    except Exception as exc:
        logger.error(
            "persona_read_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "人格设置暂时不可用，请稍后重试。")
        return
    try:
        answer = await ollama.chat(
            history,
            prompt,
            images,
            identity=identity,
            profile_targets=profile_targets,
            system_prompt=system_prompt,
        )
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
