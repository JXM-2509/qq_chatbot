"""Exercise the configured configured image model, without sending a QQ message."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import time

from qq_ollama_bot.config import get_settings
from qq_ollama_bot.conversation.identity import MemberIdentity
from qq_ollama_bot.chat.images import EncodedImage, detect_image_mime
from qq_ollama_bot.models.model import create_model_client
from qq_ollama_bot.models.model_errors import ModelError


async def check(path: Path):
    settings = get_settings()
    data = path.read_bytes()
    mime = detect_image_mime(data)
    if not mime:
        raise ValueError("Unsupported image format")
    client = create_model_client(settings)
    started = time.monotonic()
    result = {"image_count": 1, "timeout_seconds": settings.image_model_timeout_seconds}
    print(json.dumps({"status": "started", "provider": settings.model_provider, **result}), flush=True)
    try:
        answer = await client.chat(
            [], "请简短描述图片中的内容。",
            [EncodedImage(base64.b64encode(data).decode("ascii"), mime)],
            identity=MemberIdentity(group_id=0, user_id=0, nickname="本地验证",
                                    card=None, role="member", updated_at=int(time.time())),
            request_timeout_seconds=settings.image_model_timeout_seconds,
        )
        result.update(status="success", reply_characters=len(answer))
    except ModelError as exc:
        result.update(status="failed", error=type(exc).__name__)
    finally:
        await client.close()
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    print(json.dumps(result), flush=True)
    return result["status"] == "success"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    raise SystemExit(0 if asyncio.run(check(parser.parse_args().image)) else 1)
