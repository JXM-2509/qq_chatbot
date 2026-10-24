from __future__ import annotations

import nonebot
import signal
import logging
from nonebot.adapters.onebot.v11 import Adapter

from qq_ollama_bot.runtime.credentials import load_startup_settings
from qq_ollama_bot.runtime.health import install_health


def main() -> None:
    if hasattr(signal, "SIGBREAK"):
        def console_break(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, console_break)
    settings = load_startup_settings()
    # Only our sanitized model diagnostics go to the supervisor's stdout reader.
    model_logger = logging.getLogger("qq_ollama_bot.models.deepseek")
    if not model_logger.handlers:
        model_logger.addHandler(logging.StreamHandler())
    model_logger.setLevel(settings.log_level)
    model_logger.propagate = False
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
    install_health(driver, settings)
    nonebot.run()


if __name__ == "__main__":
    main()
