from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ImageInputError(ValueError):
    """An image cannot safely be sent to Ollama."""


@dataclass(frozen=True)
class EncodedImage:
    data: str
    mime_type: str


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _read_checked_image(path: Path, max_bytes: int) -> tuple[bytes, str]:
    if not path.is_absolute():
        raise ImageInputError("NapCat 返回了无效的图片路径。")
    try:
        size = await asyncio.to_thread(lambda: path.stat().st_size)
    except OSError as exc:
        raise ImageInputError("图片缓存读取失败，请重新发送图片。") from exc
    if size > max_bytes:
        raise ImageInputError(
            f"单张图片不能超过 {max_bytes / (1024 * 1024):g} MB。"
        )

    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise ImageInputError("图片缓存读取失败，请重新发送图片。") from exc
    if len(data) > max_bytes:
        raise ImageInputError(
            f"单张图片不能超过 {max_bytes / (1024 * 1024):g} MB。"
        )

    mime_type = detect_image_mime(data)
    if mime_type is None:
        raise ImageInputError("仅支持 JPEG、PNG 或 WebP 图片。")
    return data, mime_type


async def load_images(
    bot: Any,
    segments: Iterable[Any],
    *,
    max_count: int,
    max_bytes: int,
    total_max_bytes: int,
) -> list[EncodedImage]:
    image_segments = list(segments)
    if len(image_segments) > max_count:
        raise ImageInputError(f"一次最多发送 {max_count} 张图片。")

    encoded: list[EncodedImage] = []
    total_bytes = 0
    for segment in image_segments:
        file_id = segment.data.get("file")
        if not file_id:
            raise ImageInputError("图片缺少文件标识，请重新发送。")
        try:
            result = await bot.call_api("get_image", file=str(file_id))
        except Exception as exc:
            raise ImageInputError("无法从 NapCat 读取图片，请重新发送。") from exc
        if not isinstance(result, dict) or not result.get("file"):
            raise ImageInputError("NapCat 未返回可用的图片缓存。")

        data, mime_type = await _read_checked_image(Path(str(result["file"])), max_bytes)
        total_bytes += len(data)
        if total_bytes > total_max_bytes:
            raise ImageInputError(
                f"图片总大小不能超过 {total_max_bytes / (1024 * 1024):g} MB。"
            )
        encoded.append(
            EncodedImage(
                data=base64.b64encode(data).decode("ascii"),
                mime_type=mime_type,
            )
        )
    return encoded
