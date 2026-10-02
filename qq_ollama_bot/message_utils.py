from __future__ import annotations

from nonebot.adapters.onebot.v11 import Message


def has_explicit_at(message: Message, bot_id: str | int) -> bool:
    expected = str(bot_id)
    return any(
        segment.type == "at" and str(segment.data.get("qq", "")) == expected
        for segment in message
    )


def should_reply_to_group_message(
    message: Message,
    bot_id: str | int,
    *,
    to_me: bool,
) -> bool:
    return to_me or has_explicit_at(message, bot_id)


def get_at_user_ids(
    message: Message,
    *,
    exclude: set[int] | frozenset[int] = frozenset(),
) -> list[int]:
    user_ids: list[int] = []
    seen: set[int] = set()
    for segment in message:
        if segment.type != "at":
            continue
        try:
            user_id = int(str(segment.data.get("qq", "")))
        except ValueError:
            continue
        if user_id in exclude or user_id in seen:
            continue
        seen.add(user_id)
        user_ids.append(user_id)
    return user_ids


def extract_text(message: Message) -> str:
    return "".join(
        str(segment.data.get("text", ""))
        for segment in message
        if segment.type == "text"
    ).strip()


def get_image_segments(message: Message) -> list:
    return [segment for segment in message if segment.type == "image"]


def split_reply(text: str, limit: int = 1_500) -> list[str]:
    remaining = text.strip()
    if not remaining:
        return []

    chunks: list[str] = []
    preferred_breaks = ("\n", "。", "！", "？", "；", ". ", "! ", "? ")
    while len(remaining) > limit:
        window = remaining[:limit]
        minimum_break = limit // 2
        cut = max(window.rfind(marker) for marker in preferred_breaks)
        if cut < minimum_break:
            cut = limit
        elif window[cut] != "\n":
            cut += 1
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
