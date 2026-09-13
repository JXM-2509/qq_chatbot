from __future__ import annotations

import nonebot
from nonebot.adapters.onebot.v11 import Adapter

from qq_ollama_bot.config import get_settings


def main() -> None:
    settings = get_settings()
    nonebot.init(
        driver="~fastapi",
        host=settings.bot_host,
        port=settings.bot_port,
        log_level=settings.log_level,
        onebot_access_token=settings.onebot_access_token.get_secret_value(),
    )
    driver = nonebot.get_driver()
    driver.register_adapter(Adapter)
    nonebot.load_plugin("qq_ollama_bot.plugins.chat")
    nonebot.run()


if __name__ == "__main__":
    main()

