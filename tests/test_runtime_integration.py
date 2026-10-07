"""Real process smoke tests with isolated database and no QQ login or group messages."""
import os
from pathlib import Path
import socket
import sys
import time

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows integration")


def test_owned_bot_serves_health_before_qq_login(tmp_path):
    from qq_ollama_bot.windows_process import OwnedProcess
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = os.environ.copy()
    env.update(ONEBOT_ACCESS_TOKEN="integration-test-token-0123456789", ALLOWED_GROUP_IDS="1",
               DATABASE_PATH=str(tmp_path / "isolated.db"), BOT_PORT=str(port),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    root = Path(__file__).resolve().parents[1]
    child = OwnedProcess([sys.executable, "-m", "qq_ollama_bot"], root, env, lambda line: None)
    try:
        deadline = time.monotonic() + 30
        result = None
        with httpx.Client(timeout=8, trust_env=False) as client:
            while time.monotonic() < deadline:
                try:
                    reply = client.get(f"http://127.0.0.1:{port}/healthz")
                    reply.raise_for_status()
                    result = reply.json()
                    break
                except httpx.HTTPError:
                    time.sleep(0.1)
        assert result is not None and result["service"] == "qq-ollama-bot"
        assert result["ready"]
        assert not result["onebot_connected"] and result["qq_online"] is None
        assert (tmp_path / "isolated.db").is_file()
    finally:
        child.stop(grace=3)
    with httpx.Client(timeout=1, trust_env=False) as client:
        with pytest.raises(httpx.HTTPError):
            client.get(f"http://127.0.0.1:{port}/healthz")
