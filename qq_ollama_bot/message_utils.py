from __future__ import annotations

from nonebot.adapters.onebot.v11 import Message


def has_explicit_at(message: Message, bot_id: str | int) -> bool:
    expected = str(bot_id)
    return any(
        segment.type == "at" and str(segment.data.get("qq", "")) == expected
        for segment in message
    )


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

