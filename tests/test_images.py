from pathlib import Path

import pytest

from qq_ollama_bot.services.images import (
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
