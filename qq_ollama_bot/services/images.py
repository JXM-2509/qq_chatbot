from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx


logger = logging.getLogger(__name__)
QQ_IMAGE_HOSTS = frozenset({
    "gchat.qpic.cn", "c2cpicdw.qpic.cn", "multimedia.nt.qq.com.cn",
    "multimedia.nt.qq.com", "gxh.vip.qq.com",
})
IMAGE_FAILURE_LABELS = {
    "napcat_api_failed": "NapCat 图片接口失败",
    "napcat_api_timeout": "NapCat 图片接口超时",
    "no_cache": "未返回图片缓存",
    "invalid_cache_path": "图片缓存路径无效",
    "cache_unreadable": "图片缓存不可读",
    "image_too_large": "单张图片超过大小限制",
    "unsupported_image_format": "图片格式不支持",
    "total_images_too_large": "图片总大小超过限制",
    "missing_image_id": "图片缺少标识和地址",
    "invalid_base64": "图片编码无效",
    "unsafe_image_url": "图片地址不受支持",
    "image_url_failed": "图片地址下载失败或已过期",
}


class ImageInputError(ValueError):
    """An image cannot safely be sent to Ollama."""

    def __init__(self, message: str, *, code: str = "image_input") -> None:
        super().__init__(message)
        self.code = code


def image_failure_label(error: ImageInputError) -> str:
    return IMAGE_FAILURE_LABELS.get(error.code, "图片读取失败")


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
        raise ImageInputError("NapCat 返回了无效的图片路径。", code="invalid_cache_path")
    try:
        size = await asyncio.to_thread(lambda: path.stat().st_size)
    except OSError as exc:
        raise ImageInputError("图片缓存读取失败，请重新发送图片。", code="cache_unreadable") from exc
    if size > max_bytes:
        raise ImageInputError(
            f"单张图片不能超过 {max_bytes / (1024 * 1024):g} MB。", code="image_too_large",
        )

    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise ImageInputError("图片缓存读取失败，请重新发送图片。", code="cache_unreadable") from exc
    if len(data) > max_bytes:
        raise ImageInputError(
            f"单张图片不能超过 {max_bytes / (1024 * 1024):g} MB。", code="image_too_large",
        )

    mime_type = detect_image_mime(data)
    if mime_type is None:
        raise ImageInputError("仅支持 JPEG、PNG 或 WebP 图片。", code="unsupported_image_format")
    return data, mime_type


def _checked_bytes(data: bytes, max_bytes: int) -> tuple[bytes, str]:
    if len(data) > max_bytes:
        raise ImageInputError(
            f"单张图片不能超过 {max_bytes / (1024 * 1024):g} MB。", code="image_too_large",
        )
    mime_type = detect_image_mime(data)
    if mime_type is None:
        raise ImageInputError("仅支持 JPEG、PNG 或 WebP 图片。", code="unsupported_image_format")
    return data, mime_type


def _checked_image_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in QQ_IMAGE_HOSTS
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in {None, 80 if parsed.scheme == "http" else 443}
        ):
            raise ValueError("unsupported image URL")
        # Older QQ image URLs may use HTTP. Always download over HTTPS.
        return urlunsplit(("https", parsed.hostname, parsed.path, parsed.query, ""))
    except ValueError as exc:
        raise ImageInputError("图片下载地址不受支持。", code="unsafe_image_url") from exc


async def _download_image(
    url: str, max_bytes: int, transport: httpx.AsyncBaseTransport | None,
) -> tuple[bytes, str]:
    current = _checked_image_url(url)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(10, connect=5), follow_redirects=False,
        trust_env=False, transport=transport,
    ) as client:
        try:
            for _ in range(4):
                async with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            break
                        current = _checked_image_url(urljoin(current, location))
                        continue
                    response.raise_for_status()
                    try:
                        declared_size = int(response.headers.get("content-length", "0"))
                    except ValueError:
                        declared_size = 0
                    if declared_size > max_bytes:
                        raise ImageInputError("图片下载超过单张大小限制。", code="image_too_large")
                    data = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if len(data) + len(chunk) > max_bytes:
                            raise ImageInputError("图片下载超过单张大小限制。", code="image_too_large")
                        data.extend(chunk)
                    return _checked_bytes(bytes(data), max_bytes)
        except httpx.HTTPError as exc:
            # Do not put signed image URLs (including rkey) in messages or logs.
            logger.warning(
                "image_url_download_failed error_type=%s status=%s",
                type(exc).__name__, getattr(getattr(exc, "response", None), "status_code", None),
            )
            raise ImageInputError("图片地址下载失败或已过期，请重新发送。", code="image_url_failed") from exc
    raise ImageInputError("图片下载重定向次数过多。", code="image_url_failed")


async def _load_single_image(
    bot: Any, segment: Any, max_bytes: int, transport: httpx.AsyncBaseTransport | None,
) -> tuple[bytes, str]:
    file_id = segment.data.get("file")
    result: dict[str, Any] = {}
    failure = ImageInputError("图片缺少文件标识，请重新发送。", code="missing_image_id")
    if file_id:
        try:
            response = await asyncio.wait_for(
                bot.call_api("get_image", file=str(file_id)), timeout=10,
            )
            if isinstance(response, dict):
                result = response
            if result.get("file"):
                return await _read_checked_image(Path(str(result["file"])), max_bytes)
            failure = ImageInputError("NapCat 未返回可用的图片缓存。", code="no_cache")
        except ImageInputError as exc:
            if exc.code in {"image_too_large", "unsupported_image_format"}:
                raise
            failure = exc
        except asyncio.TimeoutError:
            failure = ImageInputError("NapCat 图片获取超时，请重新发送。", code="napcat_api_timeout")
        except Exception as exc:
            logger.warning("image_cache_api_failed error_type=%s", type(exc).__name__)
            failure = ImageInputError("无法从 NapCat 读取图片，请重新发送。", code="napcat_api_failed")

    logger.info(
        "image_cache_unavailable reason=%s cause_type=%s", failure.code,
        type(failure.__cause__).__name__ if failure.__cause__ is not None else "none",
    )
    encoded = result.get("base64")
    if isinstance(encoded, str) and encoded:
        if len(encoded) > (max_bytes + 2) // 3 * 4:
            raise ImageInputError("图片编码超过单张大小限制。", code="image_too_large")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            failure = ImageInputError("NapCat 返回了无效的图片编码。", code="invalid_base64")
        else:
            return _checked_bytes(decoded, max_bytes)

    # Merged-forward images often have a valid CDN URL but no local message cache.
    urls = list(dict.fromkeys(
        value for value in (result.get("url"), segment.data.get("url"))
        if isinstance(value, str) and value
    ))
    for url in urls:
        try:
            loaded = await asyncio.wait_for(_download_image(url, max_bytes, transport), timeout=15)
            logger.info("image_url_fallback_success")
            return loaded
        except ImageInputError as exc:
            if exc.code == "image_too_large":
                raise
            failure = exc
        except asyncio.TimeoutError:
            failure = ImageInputError("图片地址下载超时，请重新发送。", code="image_url_failed")
    raise failure


async def load_images(
    bot: Any,
    segments: Iterable[Any],
    *,
    max_count: int,
    max_bytes: int,
    total_max_bytes: int,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[EncodedImage]:
    image_segments = list(segments)
    if len(image_segments) > max_count:
        raise ImageInputError(f"一次最多发送 {max_count} 张图片。")

    encoded: list[EncodedImage] = []
    total_bytes = 0
    for segment in image_segments:
        try:
            data, mime_type = await _load_single_image(bot, segment, max_bytes, transport)
        except ImageInputError as exc:
            logger.warning("image_load_failed reason=%s", exc.code)
            raise
        total_bytes += len(data)
        if total_bytes > total_max_bytes:
            logger.warning("image_load_failed reason=total_images_too_large")
            raise ImageInputError(
                f"图片总大小不能超过 {total_max_bytes / (1024 * 1024):g} MB。",
                code="total_images_too_large",
            )
        encoded.append(
            EncodedImage(
                data=base64.b64encode(data).decode("ascii"),
                mime_type=mime_type,
            )
        )
    return encoded
