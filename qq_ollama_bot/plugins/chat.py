from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from dataclasses import dataclass
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
from qq_ollama_bot.services.forwards import (
    ForwardInputError, ForwardReader, get_forward_segments, prepare_forward,
)
from qq_ollama_bot.services.forward_queue import ForwardQueue
from qq_ollama_bot.services.identity import IdentityStore, MemberIdentity
from qq_ollama_bot.services.memory import MemoryStore
from qq_ollama_bot.services.persona import PersonaCatalog, PersonaSelection, PersonaStore
from qq_ollama_bot.services.ollama import (
    MemberProfileEvidence,
    OllamaBusyError,
    OllamaClient,
    OllamaError,
    FORWARD_CONTEXT_LABEL,
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
    forward_queue.start()
    logger.info(
        "services_started "
        f"expired_messages_removed={removed_messages}"
    )


@driver.on_shutdown
async def shutdown_services() -> None:
    await forward_queue.stop()
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
    parts = normalized.split()
    if normalized == "/help":
        await safe_reply(
            bot,
            event,
            "可用命令：\n"
            "/help 查看帮助\n"
            "/reset 清除你在当前群当前人格的对话记忆\n"
            "/reset all 清除你在当前群全部人格的对话记忆\n"
            "/status 检查本地模型状态\n"
            "/persona 查看当前人格；/persona list 查看可用人格\n"
            "/persona <名称> 切换你在当前群的人格\n"
            "直接发送合并转发聊天记录：自动阅读文字和图片并回应\n"
            "@机器人 @群友 问题：基于该群友的近期发言回答",
        )
        return True
    if parts and parts[0] == "/reset":
        if parts not in (["/reset"], ["/reset", "all"]):
            await safe_reply(bot, event, "用法：/reset 或 /reset all")
            return True
        try:
            if len(parts) == 2:
                await memory.reset_all_personas(event.group_id, event.user_id)
                reply = "已清除你在当前群全部人格的对话记忆。"
            else:
                name = await personas.get_active(event.group_id, event.user_id)
                await memory.reset_session(
                    event.group_id, event.user_id, persona_name=name
                )
                reply = f"已清除你在当前群 {name} 人格的对话记忆。"
        except Exception as exc:
            logger.error(
                "memory_reset_failed "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"user_id={event.user_id} error={type(exc).__name__}"
            )
            await safe_reply(bot, event, "记忆服务暂时不可用，请稍后重试。")
        else:
            await safe_reply(bot, event, reply)
        return True
    if normalized == "/status":
        healthy = await ollama.health()
        message = "机器人和本地模型均可用。" if healthy else "机器人在线，但本地模型暂时不可用。"
        await safe_reply(bot, event, message)
        return True
    if parts and parts[0] == "/persona":
        try:
            if len(parts) == 1:
                name = await personas.get_active(event.group_id, event.user_id)
                if persona_catalog.get_prompt(name) is None:
                    reply = f"当前人格：{name}（提示词不可用，请切换到可用人格）"
                else:
                    reply = f"当前人格：{name}"
            elif len(parts) == 2 and parts[1] == "list":
                reply = "可用人格：" + "、".join(persona_catalog.names())
            elif len(parts) == 2:
                name = parts[1]
                if persona_catalog.get_prompt(name) is None:
                    reply = "没有这个人格。发送 /persona list 查看可用人格。"
                else:
                    await personas.set_active(event.group_id, event.user_id, name)
                    reply = (
                        f"已切换为 {name}，仅影响你在当前群之后的新请求，"
                        "使用该人格的独立对话记忆。"
                    )
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


@dataclass(frozen=True)
class ForwardRequest:
    bot: Bot
    event: GroupMessageEvent
    identity: MemberIdentity
    selection: PersonaSelection
    system_prompt: str


async def read_request_persona(
    bot: Bot, event: GroupMessageEvent,
) -> tuple[PersonaSelection, str] | None:
    try:
        selection = await personas.get_selection(event.group_id, event.user_id)
        prompt = persona_catalog.get_prompt(selection.name)
        if prompt is None:
            await safe_reply(
                bot, event,
                f"当前人格 {selection.name} 的提示词不可用，请用 /persona list 查看并切换人格。",
            )
            return None
        return selection, prompt
    except Exception as exc:
        logger.error(
            f"persona_read_failed event_id={event.message_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "人格设置暂时不可用，请稍后重试。")
        return None


async def finish_reply(
    bot: Bot, event: GroupMessageEvent, selection: PersonaSelection,
    answer: str, memory_text: str,
) -> None:
    reply = answer
    try:
        current_selection = await personas.get_selection(event.group_id, event.user_id)
        if current_selection != selection:
            reply = f"【这是切换前的 {selection.name} 人格请求回复】\n{answer}"
    except Exception as exc:
        logger.warning(
            f"persona_reply_check_failed event_id={event.message_id} error={type(exc).__name__}"
        )
        reply = f"【本回复来自请求开始时的 {selection.name} 人格】\n{answer}"
    await safe_reply(bot, event, reply)
    try:
        await memory.append_exchange(
            event.group_id, event.user_id, memory_text, answer,
            persona_name=selection.name,
        )
    except Exception as exc:
        logger.error(
            f"memory_write_failed event_id={event.message_id} error={type(exc).__name__}"
        )


async def process_forward_request(request: ForwardRequest) -> None:
    bot, event = request.bot, request.event
    started_at = time.monotonic()
    reader = ForwardReader(
        bot, max_depth=settings.forward_max_depth,
        max_messages=settings.forward_message_limit,
        character_budget=settings.forward_character_budget,
        timeout_seconds=settings.forward_timeout_seconds,
    )
    evidence = await reader.read(get_forward_segments(event.original_message))
    try:
        prepared = await prepare_forward(
            bot, evidence, max_images=settings.image_max_count,
            image_max_bytes=settings.image_max_bytes,
            image_total_max_bytes=settings.image_total_max_bytes,
            timeout_seconds=settings.forward_timeout_seconds,
        )
    except ForwardInputError as exc:
        await safe_reply(bot, event, str(exc))
        return
    try:
        history = await memory.get_history(
            event.group_id, event.user_id, persona_name=request.selection.name,
        )
    except Exception as exc:
        logger.error(f"memory_read_failed event_id={event.message_id} error={type(exc).__name__}")
        await safe_reply(bot, event, "记忆服务暂时不可用，请稍后重试。")
        return
    question = extract_text(event.message)
    prompt = question or "请阅读这份合并转发聊天记录，沿用当前人格自然简短回应其中的内容。"
    fingerprint = hashlib.sha256(request.system_prompt.encode("utf-8")).hexdigest()[:12]
    logger.info(
        f"forward_request_started event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} persona_name={request.selection.name} "
        f"persona_revision={request.selection.revision} prompt_fingerprint={fingerprint} "
        f"message_count={reader.message_count} image_count={len(prepared.images)} "
        f"history_count={len(history)}"
    )
    try:
        answer = await ollama.chat(
            history, prompt, prepared.images, identity=request.identity,
            system_prompt=request.system_prompt, forwarded_context=prepared.context,
            wait_when_busy=True,
        )
    except OllamaError as exc:
        logger.error(f"forward_model_failed event_id={event.message_id} error={type(exc).__name__}")
        await safe_reply(bot, event, "本地模型暂时不可用，请稍后重新发送记录。")
        return
    # Reading limitations are program-generated; do not rely on the model to report them.
    if prepared.notes:
        answer = "【读取说明】" + "；".join(prepared.notes) + "\n" + answer
    memory_text = f"{FORWARD_CONTEXT_LABEL}\n{prepared.context}\n[用户消息]\n{prompt}"
    await finish_reply(bot, event, request.selection, answer, memory_text)
    logger.info(
        f"forward_request_completed event_id={event.message_id} "
        f"elapsed_ms={int((time.monotonic() - started_at) * 1000)}"
    )


async def forward_request_failed(request: ForwardRequest, exc: Exception) -> None:
    logger.error(
        f"forward_request_failed event_id={request.event.message_id} error={type(exc).__name__}"
    )
    await safe_reply(request.bot, request.event, "读取转发记录时发生错误，请稍后重新发送。")


forward_queue = ForwardQueue(
    process_forward_request, forward_request_failed, max_pending=settings.forward_queue_size,
)


@chat_matcher.handle()
async def handle_group_message(bot: Bot, event: GroupMessageEvent) -> None:
    if event.group_id not in settings.allowed_group_id_set or event.user_id == int(event.self_id):
        return
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
    forwards = get_forward_segments(original_message)
    original_text = extract_text(original_message)
    mentioned_user_ids = get_at_user_ids(original_message)
    try:
        inserted = await group_messages.append_event(
            group_id=event.group_id,
            message_id=event.message_id,
            user_id=event.user_id,
            content=original_text,
            created_at=event.time,
            mentioned_user_ids=mentioned_user_ids,
        )
        if forwards and not inserted:
            return
    except Exception as exc:
        logger.error(
            "group_message_archive_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )

    if forwards:
        text = extract_text(event.message)
        if len(text) > 4_000:
            await safe_reply(bot, event, "附带问题不能超过 4000 个字符。")
            return
        persona = await read_request_persona(bot, event)
        if persona is None:
            return
        selection, system_prompt = persona
        outcome = forward_queue.enqueue(
            event_id, ForwardRequest(bot, event, identity, selection, system_prompt),
        )
        if outcome == "full":
            await safe_reply(bot, event, "转发记录等待队列已满，请稍后重新发送。")
        elif outcome == "closed":
            await safe_reply(bot, event, "转发阅读服务暂时不可用，请稍后重新发送。")
        return

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

    try:
        persona_selection = await personas.get_selection(event.group_id, event.user_id)
        persona_name = persona_selection.name
        system_prompt = persona_catalog.get_prompt(persona_name)
        if system_prompt is None:
            logger.warning(
                "persona_unavailable "
                f"event_id={event.message_id} group_id={event.group_id} "
                f"user_id={event.user_id} persona_name={persona_name}"
            )
            await safe_reply(
                bot, event,
                f"当前人格 {persona_name} 的提示词不可用，请用 /persona list 查看并切换人格。",
            )
            return
    except Exception as exc:
        logger.error(
            "persona_read_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "人格设置暂时不可用，请稍后重试。")
        return

    prompt_fingerprint = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
    started_at = time.monotonic()
    logger.info(
        "request_started "
        f"event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} image_count={len(image_segments)} "
        f"persona_name={persona_name} persona_revision={persona_selection.revision} "
        f"prompt_fingerprint={prompt_fingerprint}"
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
        history = await memory.get_history(
            event.group_id, event.user_id, persona_name=persona_name
        )
    except Exception as exc:
        logger.error(
            "memory_read_failed "
            f"event_id={event.message_id} group_id={event.group_id} "
            f"user_id={event.user_id} error={type(exc).__name__}"
        )
        await safe_reply(bot, event, "记忆服务暂时不可用，请稍后重试。")
        return

    logger.info(
        "request_context "
        f"event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} persona_name={persona_name} "
        f"history_count={len(history)} prompt_fingerprint={prompt_fingerprint}"
    )
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

    memory_text = prompt
    if images:
        memory_text = f"{prompt}\n[用户发送了 {len(images)} 张图片]"
    await finish_reply(bot, event, persona_selection, answer, memory_text)

    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    logger.info(
        "request_completed "
        f"event_id={event.message_id} group_id={event.group_id} "
        f"user_id={event.user_id} persona_name={persona_name} elapsed_ms={elapsed_ms}"
    )
