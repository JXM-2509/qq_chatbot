from pathlib import Path
import base64
import logging
from types import SimpleNamespace

import httpx
import pytest

from qq_ollama_bot.chat.images import (
    ImageInputError,
    detect_image_mime,
    load_images,
)


class Segment:
    def __init__(self, file_id: str) -> None:
        self.data = {"file": file_id, "url": "https://example.invalid/must-not-fetch"}


class FakeBot:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.calls: list[tuple[str, str]] = []

    async def call_api(self, api: str, *, file: str) -> dict[str, str]:
        self.calls.append((api, file))
        return {"file": str(self.image_path.resolve())}


def test_detect_supported_image_types() -> None:
    assert detect_image_mime(b"\xff\xd8\xffrest") == "image/jpeg"
    assert detect_image_mime(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert detect_image_mime(b"RIFF\x00\x00\x00\x00WEBPrest") == "image/webp"


def test_reject_unknown_image_type() -> None:
    assert detect_image_mime(b"not-an-image") is None


@pytest.mark.asyncio
async def test_load_image_uses_onebot_cache(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\ncontent")
    bot = FakeBot(image_path)

    images = await load_images(
        bot,
        [Segment("napcat-file-id")],
        max_count=4,
        max_bytes=1024,
        total_max_bytes=2048,
    )

    assert bot.calls == [("get_image", "napcat-file-id")]
    assert images[0].mime_type == "image/png"


@pytest.mark.asyncio
async def test_reject_image_before_reading_oversized_file(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 100)
    bot = FakeBot(image_path)

    with pytest.raises(ImageInputError, match="单张图片不能超过"):
        await load_images(
            bot,
            [Segment("large")],
            max_count=4,
            max_bytes=10,
            total_max_bytes=20,
        )


class CacheMissBot:
    def __init__(self, response=None):
        self.response = response

    async def call_api(self, api, *, file):
        if self.response is None:
            raise RuntimeError("element not found")
        return self.response


async def load_with_fallback(bot, segments, transport, **limits):
    return await load_images(bot, segments, transport=transport, **{
        "max_count": 4, "max_bytes": 1024, "total_max_bytes": 2048, **limits,
    })


def url_segment(url, file="forward-image-file"):
    return SimpleNamespace(data={"file": file, "url": url})


@pytest.mark.asyncio
async def test_forward_cache_miss_downloads_qq_url_without_logging_signed_query(caplog):
    payload = b"\x89PNG\r\n\x1a\ncontent"
    urls = []

    async def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, content=payload)

    url = "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=forward&rkey=test-secret"
    with caplog.at_level(logging.INFO, logger="qq_ollama_bot.chat.images"):
        images = await load_with_fallback(CacheMissBot(), [url_segment(url)], httpx.MockTransport(handler))
    assert base64.b64decode(images[0].data) == payload
    assert urls == [url]
    assert "image_url_fallback_success" in caplog.text
    assert "test-secret" not in caplog.text and url not in caplog.text


@pytest.mark.asyncio
async def test_unreadable_cache_path_falls_back_to_segment_url(tmp_path):
    async def handler(request):
        return httpx.Response(200, content=b"\xff\xd8\xffimage")

    images = await load_with_fallback(
        CacheMissBot({"file": str(tmp_path / "missing.jpg")}),
        [url_segment("https://gchat.qpic.cn/image")], httpx.MockTransport(handler),
    )
    assert images[0].mime_type == "image/jpeg"


@pytest.mark.asyncio
async def test_napcat_base64_and_url_only_segment_are_supported():
    payload = b"\x89PNG\r\n\x1a\ncontent"

    async def handler(request):
        return httpx.Response(200, content=payload)

    encoded = base64.b64encode(payload).decode("ascii")
    images = await load_with_fallback(CacheMissBot({"base64": encoded}), [Segment("image")], None)
    assert images[0].data == encoded
    images = await load_with_fallback(
        CacheMissBot(), [url_segment("https://gchat.qpic.cn/image", file="")], httpx.MockTransport(handler),
    )
    assert images[0].data == encoded


@pytest.mark.asyncio
async def test_returned_url_failure_can_use_original_segment_url():
    urls = []

    async def handler(request):
        urls.append(str(request.url))
        if request.url.path == "/expired":
            return httpx.Response(403)
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\ncontent")

    images = await load_with_fallback(
        CacheMissBot({"url": "https://gchat.qpic.cn/expired"}),
        [url_segment("http://gchat.qpic.cn/original")], httpx.MockTransport(handler),
    )
    assert len(images) == 1
    assert urls == ["https://gchat.qpic.cn/expired", "https://gchat.qpic.cn/original"]


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/private", "file:///C:/private.png", "https://example.com/image",
    "https://gchat.qpic.cn.example.com/image", "https://user:pass@gchat.qpic.cn/image",
    "https://gchat.qpic.cn:8443/image",
])
async def test_unknown_hosts_and_local_urls_are_never_downloaded(url):
    async def handler(request):
        pytest.fail("Rejected URL must not issue a network request")

    with pytest.raises(ImageInputError) as failure:
        await load_with_fallback(CacheMissBot(), [url_segment(url)], httpx.MockTransport(handler))
    assert failure.value.code == "unsafe_image_url"


@pytest.mark.asyncio
async def test_redirect_to_non_qq_host_is_rejected():
    calls = []

    async def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    with pytest.raises(ImageInputError) as failure:
        await load_with_fallback(CacheMissBot(), [url_segment("https://gchat.qpic.cn/image")], httpx.MockTransport(handler))
    assert failure.value.code == "unsafe_image_url" and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [True, False])
async def test_url_download_enforces_size_limit(header):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\x89PNG\r\n\x1a\n" + b"x" * 1100

    async def handler(request):
        return httpx.Response(200, headers={"content-length": "1108"} if header else {}, stream=Stream())

    with pytest.raises(ImageInputError) as failure:
        await load_with_fallback(CacheMissBot(), [url_segment("https://gchat.qpic.cn/image")], httpx.MockTransport(handler))
    assert failure.value.code == "image_too_large"


@pytest.mark.asyncio
@pytest.mark.parametrize("status,payload,code", [
    (403, b"", "image_url_failed"), (200, b"<html>not an image</html>", "unsupported_image_format"),
    (200, b"GIF89aimage", "unsupported_image_format"),
])
async def test_expired_urls_non_images_and_gifs_have_distinct_reasons(status, payload, code):
    async def handler(request):
        return httpx.Response(status, content=payload)

    with pytest.raises(ImageInputError) as failure:
        await load_with_fallback(CacheMissBot(), [url_segment("https://gchat.qpic.cn/image")], httpx.MockTransport(handler))
    assert failure.value.code == code
