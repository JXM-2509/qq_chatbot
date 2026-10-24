from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence

import httpx

from qq_ollama_bot.chat.images import EncodedImage
from qq_ollama_bot.conversation.identity import MemberIdentity

from qq_ollama_bot.models.model_common import (
    SYSTEM_PROMPT, FORWARD_CONTEXT_RULES, FORWARD_CONTEXT_LABEL,
    MemberProfileEvidence, build_current_prompt,
)
from qq_ollama_bot.models.model_errors import (
    ModelError as OllamaError, ModelBusyError as OllamaBusyError,
    ModelTimeoutError as OllamaTimeoutError, ModelContextError as OllamaContextError,
)

logger = logging.getLogger(__name__)
CONTEXT_TOKENS = 8192
MAX_CONTEXT_TOKENS = 32768


def required_context_tokens(response: httpx.Response) -> int | None:
    """Recognize only Ollama's context overflow; never log its raw error body."""
    if response.status_code != 400:
        return None
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None
    if not isinstance(error, str):
        return None
    match = re.search(
        r"request \(([0-9]{1,9}) tokens\) exceeds the available context size \([0-9]{1,9} tokens\)",
        error,
    )
    return int(match[1]) if match else None


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        request_timeout_seconds: float,
        busy_wait_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._busy_wait_seconds = busy_wait_seconds
        self._semaphore = asyncio.Semaphore(1)
        timeout = httpx.Timeout(request_timeout_seconds, connect=5.0)
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get("/api/version")
            response.raise_for_status()
            return bool(response.json().get("version"))
        except (httpx.HTTPError, ValueError, AttributeError):
            return False

    async def chat(
        self,
        history: Sequence[dict[str, str]],
        prompt: str,
        images: Sequence[EncodedImage] = (),
        *,
        identity: MemberIdentity,
        profile_targets: Sequence[MemberProfileEvidence] = (),
        system_prompt: str = SYSTEM_PROMPT,
        forwarded_context: str | None = None,
        wait_when_busy: bool = False,
        request_timeout_seconds: float | None = None,
        allow_context_timeout_extension: bool = False,
    ) -> str:
        try:
            if wait_when_busy:
                await self._semaphore.acquire()
            else:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=self._busy_wait_seconds
                )
        except asyncio.TimeoutError as exc:
            raise OllamaBusyError("Ollama inference slot is busy") from exc

        try:
            current_message: dict[str, object] = {
                "role": "user",
                "content": build_current_prompt(prompt, identity, profile_targets, forwarded_context),
            }
            if images:
                current_message["images"] = [image.data for image in images]
            if forwarded_context is not None or any(
                FORWARD_CONTEXT_LABEL in item["content"] for item in history
            ):
                system_prompt = f"{system_prompt}\n\n{FORWARD_CONTEXT_RULES}"
            messages: list[dict[str, object]] = [
                {"role": "system", "content": system_prompt},
                *history,
                current_message,
            ]
            payload = {
                "model": self._model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": "10m",
                "options": {
                    "num_ctx": CONTEXT_TOKENS,
                    "num_predict": 800,
                    "temperature": 0.7,
                },
            }
            try:
                request_options = {}
                if request_timeout_seconds is not None:
                    request_options["timeout"] = httpx.Timeout(request_timeout_seconds, connect=5.0)
                for attempt in range(2):
                    response = await self._client.post("/api/chat", json=payload, **request_options)
                    required = required_context_tokens(response)
                    if required is None:
                        response.raise_for_status()
                        break
                    # Include room for the complete reply and chat template; images
                    # are counted by the model rather than estimated from byte size.
                    reserved_tokens = payload["options"]["num_predict"] + 256
                    expanded = ((required + reserved_tokens + 4095) // 4096) * 4096
                    if (attempt or expanded > MAX_CONTEXT_TOKENS
                            or expanded <= payload["options"]["num_ctx"]):
                        raise OllamaContextError("Input exceeds the bounded model context")
                    payload["options"]["num_ctx"] = expanded
                    if allow_context_timeout_extension:
                        base_timeout = (request_timeout_seconds if request_timeout_seconds is not None
                                        else self._client.timeout.read)
                        if base_timeout is not None:
                            retry_timeout = max(base_timeout, min(1200.0, base_timeout * required / CONTEXT_TOKENS))
                            request_options["timeout"] = httpx.Timeout(retry_timeout, connect=5.0)
                    logger.warning("model_context_expanded required_tokens=%d context_tokens=%d", required, expanded)
                content = response.json()["message"]["content"]
            except httpx.TimeoutException as exc:
                raise OllamaTimeoutError("Ollama request timed out") from exc
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                raise OllamaError("Ollama request failed") from exc
            if not isinstance(content, str) or not content.strip():
                raise OllamaError("Ollama returned an empty response")
            return content.strip()
        finally:
            self._semaphore.release()
