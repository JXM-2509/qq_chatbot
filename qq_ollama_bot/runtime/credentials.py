"""Load startup settings and optionally request an ephemeral API key."""
import getpass
import os
import sys
import warnings

from pydantic import ValidationError

from qq_ollama_bot.config import Settings, get_settings


def load_startup_settings(*, interactive: bool = True) -> Settings:
    try:
        return get_settings()
    except ValidationError as exc:
        missing_key = any(
            error["loc"] == ()
            and str(error.get("ctx", {}).get("error", ""))
            == "DEEPSEEK_API_KEY is required for the DeepSeek provider"
            for error in exc.errors()
        )
        if not missing_key:
            raise

    if not interactive or not sys.stdin.isatty():
        raise RuntimeError("缺少 DEEPSEEK_API_KEY；请通过环境变量提供，或在终端正常启动项目后隐藏输入。")
    try:
        # Refuse getpass's fallback to echoed input when no secure console exists.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("DeepSeek API Key（输入隐藏，仅本次运行有效）: ").strip()
    except (EOFError, getpass.GetPassWarning):
        raise RuntimeError("无法隐藏输入 API Key，请在交互终端启动，或通过环境变量提供。") from None
    if not key:
        raise RuntimeError("未输入 DeepSeek API Key，项目未启动。")
    # Child bot processes inherit this value during recovery; never write it to disk.
    os.environ["DEEPSEEK_API_KEY"] = key
    get_settings.cache_clear()
    return get_settings()
