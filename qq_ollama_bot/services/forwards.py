from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Sequence

from nonebot.adapters.onebot.v11 import Message, MessageSegment

from qq_ollama_bot.services.images import EncodedImage, ImageInputError, load_images


class ForwardInputError(ValueError):
    """No readable material was obtained from a merged forward."""


@dataclass(frozen=True)
class ForwardPart:
    text: str = ""
    image: MessageSegment | None = None
    readable_text: bool = True


@dataclass(frozen=True)
class ForwardEntry:
    sender: str
    sent_at: int | None
    parts: tuple[ForwardPart, ...]


@dataclass(frozen=True)
class ForwardEvidence:
    entries: tuple[ForwardEntry, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PreparedForward:
    context: str
    images: tuple[EncodedImage, ...]
    notes: tuple[str, ...]


def get_forward_segments(message: Message) -> list[MessageSegment]:
    # Only outer forwards can trigger a request. Quoted commands and @s are data.
    return [segment for segment in message if segment.type == "forward"]


def _segments(value: Any) -> Message:
    if isinstance(value, (str, Message)):
        message = Message(value)
    elif isinstance(value, list):
        message = Message()
        for item in value:
            if isinstance(item, MessageSegment):
                message.append(item)
            elif isinstance(item, str):
                message.extend(Message(item))
            elif isinstance(item, dict) and isinstance(item.get("type"), str):
                message.append(MessageSegment(item["type"], item.get("data", {})))
            else:
                raise ValueError("invalid message segment")
    else:
        raise ValueError("invalid message content")
    if any(not isinstance(segment.data, dict) for segment in message):
        raise ValueError("invalid segment data")
    return message


class ForwardReader:
    """One reader per request, with shared limits across all outer forwards."""

    def __init__(
        self, bot: Any, *, max_depth: int, max_messages: int,
        character_budget: int, timeout_seconds: float,
    ) -> None:
        self.bot = bot
        self.max_depth = max_depth
        self.max_messages = max_messages
        self.remaining = character_budget
        self.character_budget = character_budget
        self.timeout = timeout_seconds
        self.entries: list[ForwardEntry] = []
        self.notes: list[str] = []
        self.message_count = 0
        self.expansion_count = 0
        self.active: set[tuple[str, str | int]] = set()

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def exhausted(self) -> bool:
        if self.remaining <= 0:
            self.note("转发文本超过字符预算，后续内容已截断。")
            return True
        if self.message_count >= self.max_messages:
            self.note("转发消息超过条数限制，后续内容已截断。")
            return True
        return False

    async def read(self, forwards: Sequence[MessageSegment]) -> ForwardEvidence:
        for segment in forwards:
            if self.exhausted():
                break
            await self.expand(segment, depth=1)
        # Include JSON escaping, sender labels and image placeholders in the budget.
        # Reserve room for the fixed image-failure notes that preparation may add.
        reserved_notes = list(self.notes)
        for note in (
            "转发文本超过字符预算，后续内容已截断。",
            "部分图片超过数量限制，未读取。",
            "部分图片无法读取，回复不能依据这些图片。",
        ):
            if note not in reserved_notes:
                reserved_notes.append(note)
        envelope = len(json.dumps(
            {"messages": [], "reading_notes": reserved_notes},
            ensure_ascii=False, separators=(",", ":"),
        )) - 2
        limit = max(0, self.character_budget - envelope)
        selected: list[ForwardEntry] = []

        def fits(entry: ForwardEntry) -> bool:
            payload = [
                {"sender": item.sender, "sent_at_unix": item.sent_at, "content": [
                    {"text": part.text} if part.image is None else
                    {"image": "未读取：图片不可用、超时或超过大小限制"}
                    for part in item.parts
                ]}
                for item in [*selected, entry]
            ]
            return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= limit

        for entry in self.entries:
            if fits(entry):
                selected.append(entry)
                continue
            parts: list[ForwardPart] = []
            for part in entry.parts:
                candidate = ForwardEntry(entry.sender, entry.sent_at, tuple([*parts, part]))
                if fits(candidate):
                    parts.append(part)
                    continue
                if part.image is None:
                    low, high = 0, len(part.text)
                    while low < high:
                        middle = (low + high + 1) // 2
                        shorter = ForwardPart(text=part.text[:middle], readable_text=part.readable_text)
                        if fits(ForwardEntry(entry.sender, entry.sent_at, tuple([*parts, shorter]))):
                            low = middle
                        else:
                            high = middle - 1
                    if low:
                        parts.append(ForwardPart(text=part.text[:low], readable_text=part.readable_text))
                break
            if parts:
                selected.append(ForwardEntry(entry.sender, entry.sent_at, tuple(parts)))
            self.note("转发文本超过字符预算，后续内容已截断。")
            break
        return ForwardEvidence(tuple(selected), tuple(self.notes))

    async def expand(self, segment: MessageSegment, *, depth: int) -> None:
        if self.exhausted():
            return
        if depth > self.max_depth:
            self.note("嵌套转发超过层数限制，内层内容未读取。")
            return
        if self.expansion_count >= self.max_messages:
            self.note("转发展开次数超过限制，后续内容未读取。")
            return
        self.expansion_count += 1
        data = segment.data
        if not isinstance(data, dict):
            self.note("部分转发节点格式无效，已跳过。")
            return
        ref = data.get("id") or data.get("res_id")
        key = ("forward", str(ref)) if ref else ("object", id(data))
        if key in self.active:
            self.note("发现循环嵌套转发，循环部分未读取。")
            return
        self.active.add(key)
        try:
            nodes = data.get("content")
            if not isinstance(nodes, list) or not nodes:
                if not ref:
                    self.note("转发缺少可用标识或内容，无法读取该部分。")
                    return
                try:
                    result = await asyncio.wait_for(
                        self.bot.call_api("get_forward_msg", message_id=str(ref)),
                        timeout=self.timeout,
                    )
                    nodes = result.get("messages") if isinstance(result, dict) else None
                except asyncio.TimeoutError:
                    self.note("转发获取超时，该部分未读取。")
                    return
                except Exception:
                    self.note("转发记录可能已过期或接口不可用，该部分未读取。")
                    return
            if not isinstance(nodes, list) or not nodes:
                self.note("转发详情为空或格式无效，该部分未读取。")
                return
            for node in nodes:
                if self.exhausted():
                    break
                await self.visit_node(node, depth=depth)
        finally:
            self.active.remove(key)

    async def visit_node(self, node: Any, *, depth: int) -> None:
        if isinstance(node, MessageSegment):
            node = {"type": node.type, "data": node.data}
        # Failed/invalid nodes also consume a slot, bounding get_msg API attempts.
        if not isinstance(node, dict):
            self.message_count += 1
            self.note("部分转发节点格式无效，已跳过。")
            return
        if node.get("type") == "forward":
            await self.expand(MessageSegment("forward", node.get("data", {})), depth=depth + 1)
            return
        self.message_count += 1
        if node.get("type") == "node":
            data = node.get("data")
        else:
            # NapCat returns complete message objects; other implementations use nodes.
            data = node
        if not isinstance(data, dict):
            self.note("部分转发节点格式无效，已跳过。")
            return
        key = ("node", str(data["id"])) if data.get("id") else ("object", id(data))
        if key in self.active:
            self.note("发现循环嵌套节点，循环部分未读取。")
            return
        self.active.add(key)
        try:
            content = data.get("message")
            if content is None or content == []:
                content = data.get("content")
            if content is None and data.get("id"):
                try:
                    data = await asyncio.wait_for(
                        self.bot.call_api("get_msg", message_id=int(data["id"])),
                        timeout=self.timeout,
                    )
                    content = data.get("message")
                except Exception:
                    self.note("引用的转发节点无法读取，已跳过。")
                    return
            try:
                message = _segments(content)
            except (ValueError, TypeError, KeyError):
                self.note("部分转发消息内容格式无效，已跳过。")
                return
            sender = data.get("sender")
            sender = sender if isinstance(sender, dict) else {}
            name = sender.get("card") or sender.get("nickname") or data.get("nickname")
            name = " ".join(str(name or "未知发送者").split())[:100]
            try:
                timestamp = int(data.get("time"))
                if timestamp > 100_000_000_000:
                    timestamp //= 1000
                if not 0 <= timestamp <= 253_402_300_799:
                    timestamp = None
            except (ValueError, TypeError, OverflowError):
                timestamp = None
            parts: list[ForwardPart] = []

            def flush() -> None:
                if parts:
                    self.entries.append(ForwardEntry(name, timestamp, tuple(parts)))
                    parts.clear()

            for segment in message:
                if self.remaining <= 0:
                    self.note("转发文本超过字符预算，后续内容已截断。")
                    break
                if segment.type in {"forward", "node"}:
                    flush()
                    if segment.type == "forward":
                        await self.expand(segment, depth=depth + 1)
                    elif depth >= self.max_depth:
                        self.note("嵌套节点超过层数限制，内层内容未读取。")
                    elif not self.exhausted():
                        await self.visit_node({"type": "node", "data": segment.data}, depth=depth + 1)
                    continue
                if segment.type == "image":
                    if self.remaining < 24:
                        self.remaining = 0
                        self.note("转发文本超过字符预算，后续内容已截断。")
                        break
                    parts.append(ForwardPart(image=segment))
                    self.remaining -= 24
                    continue
                if segment.type == "text":
                    text = str(segment.data.get("text", ""))
                else:
                    label = {
                        "at": "@ 提及", "reply": "引用", "record": "语音",
                        "voice": "语音", "video": "视频", "file": "文件",
                        "face": "表情", "json": "卡片", "xml": "卡片",
                    }.get(segment.type, "非文字消息")
                    text = f"[{label}，未解析内容]"
                kept = text[:self.remaining]
                self.remaining -= len(kept)
                if kept:
                    parts.append(ForwardPart(text=kept, readable_text=segment.type == "text"))
                if len(kept) < len(text):
                    self.note("转发文本超过字符预算，后续内容已截断。")
            flush()
        finally:
            self.active.remove(key)


async def prepare_forward(
    bot: Any, evidence: ForwardEvidence, *, max_images: int,
    image_max_bytes: int, image_total_max_bytes: int, timeout_seconds: float,
) -> PreparedForward:
    images: list[EncodedImage] = []
    notes = list(evidence.notes)
    messages = []
    attempts = 0
    used_bytes = 0
    readable = False
    for entry in evidence.entries:
        content: list[Any] = []
        for part in entry.parts:
            if part.image is None:
                content.append({"text": part.text})
                # Placeholders for unsupported media are not readable evidence.
                if part.text.strip() and part.readable_text:
                    readable = True
                continue
            attempts += 1
            if attempts > max_images:
                content.append({"image": "未读取：超过图片数量限制"})
                if "部分图片超过数量限制，未读取。" not in notes:
                    notes.append("部分图片超过数量限制，未读取。")
                continue
            try:
                loaded = await asyncio.wait_for(
                    load_images(
                        bot, [part.image], max_count=1, max_bytes=image_max_bytes,
                        total_max_bytes=image_total_max_bytes - used_bytes,
                    ),
                    timeout=timeout_seconds,
                )
            except (ImageInputError, asyncio.TimeoutError):
                content.append({"image": "未读取：图片不可用、超时或超过大小限制"})
                if "部分图片无法读取，回复不能依据这些图片。" not in notes:
                    notes.append("部分图片无法读取，回复不能依据这些图片。")
                continue
            image = loaded[0]
            used_bytes += len(image.data) // 4 * 3 - image.data.count("=")
            images.append(image)
            content.append({"image": len(images), "status": "已附图，可识别"})
            readable = True
        messages.append({"sender": entry.sender, "sent_at_unix": entry.sent_at, "content": content})
    if not readable:
        reason = "；".join(notes) or "记录没有可读取的文字或图片。"
        raise ForwardInputError(f"无法读取这份合并转发聊天记录：{reason}")
    context = json.dumps(
        {"messages": messages, "reading_notes": notes},
        ensure_ascii=False, separators=(",", ":"),
    )
    return PreparedForward(context, tuple(images), tuple(notes))
