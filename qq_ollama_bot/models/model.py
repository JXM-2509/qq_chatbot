from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from qq_ollama_bot.config import Settings
from qq_ollama_bot.conversation.identity import MemberIdentity
from qq_ollama_bot.chat.images import EncodedImage
from qq_ollama_bot.models.model_common import SYSTEM_PROMPT, MemberProfileEvidence


class ModelClient(Protocol):
    async def close(self) -> None: ...
    async def health(self) -> bool: ...
    async def chat(
        self, history: Sequence[dict[str, str]], prompt: str,
        images: Sequence[EncodedImage] = (), *, identity: MemberIdentity,
        profile_targets: Sequence[MemberProfileEvidence] = (),
        system_prompt: str = SYSTEM_PROMPT, forwarded_context: str | None = None,
        wait_when_busy: bool = False, request_timeout_seconds: float | None = None,
        allow_context_timeout_extension: bool = False,
    ) -> str: ...


def create_model_client(settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None) -> ModelClient:
    options = dict(request_timeout_seconds=settings.request_timeout_seconds,
                   busy_wait_seconds=settings.busy_wait_seconds, transport=transport)
    if settings.model_provider == "deepseek":
        from qq_ollama_bot.models.deepseek import DeepSeekClient
        assert settings.deepseek_api_key is not None
        return DeepSeekClient(base_url=settings.deepseek_base_url, model=settings.deepseek_model,
                              api_key=settings.deepseek_api_key.get_secret_value(),
                              max_tokens=settings.deepseek_max_tokens, **options)
    from qq_ollama_bot.models.ollama import OllamaClient
    return OllamaClient(base_url=settings.ollama_base_url, model=settings.ollama_model, **options)
