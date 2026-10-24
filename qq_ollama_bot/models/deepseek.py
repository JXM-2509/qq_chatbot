from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence

import httpx

from qq_ollama_bot.conversation.identity import MemberIdentity
from qq_ollama_bot.chat.images import EncodedImage
from qq_ollama_bot.models.model_common import (
    SYSTEM_PROMPT, FORWARD_CONTEXT_LABEL, FORWARD_CONTEXT_RULES,
    MemberProfileEvidence, build_current_prompt,
)
from qq_ollama_bot.models.model_errors import (
    ModelError, ModelBusyError, ModelTimeoutError, ModelContextError,
    ModelAuthError, ModelBalanceError, ModelRateLimitError,
)

logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 48 * 1024 * 1024


class DeepSeekProbe:
    """Read-only model metadata, cached including failures; never run inference."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, api_key: str, model: str):
        self.client, self.base_url, self.api_key, self.model = client, base_url.rstrip("/"), api_key, model
        self._cached: dict | None = None
        self._expires = 0.0
        self._lock = asyncio.Lock()

    @property
    def supports_images(self) -> bool | None:
        if self._cached is None or time.monotonic() >= self._expires:
            return None
        return self._cached["model_supports_images"]

    async def snapshot(self) -> dict:
        async with self._lock:
            if self._cached is not None and time.monotonic() < self._expires:
                return dict(self._cached)
            state = {"model_provider": "deepseek", "model_name": self.model,
                     "ollama_reachable": None, "model_reachable": False,
                     "model_authenticated": None, "model_available": None,
                     "model_supports_images": None, "model_error": None}
            try:
                async with asyncio.timeout(3):
                    response = await self.client.get(
                        f"{self.base_url}/models", timeout=3,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                state["model_reachable"] = True
                if response.status_code in (401, 403):
                    state.update(model_authenticated=False, model_error="auth")
                elif response.status_code == 402:
                    state["model_error"] = "balance"
                elif response.status_code == 429:
                    state["model_error"] = "rate_limit"
                else:
                    response.raise_for_status()
                    models = response.json()["data"]
                    if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
                        raise ValueError("invalid model list")
                    model = next((item for item in models if item.get("id") == self.model), None)
                    state.update(model_authenticated=True, model_available=model is not None)
                    if model is not None:
                        modalities = model.get("input_modalities")
                        if isinstance(modalities, list):
                            state["model_supports_images"] = "image" in modalities
                    else:
                        state["model_error"] = "model_missing"
            except (httpx.TimeoutException, TimeoutError):
                state["model_error"] = "timeout"
            except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
                state["model_error"] = "unavailable"
            self._cached = state
            self._expires = time.monotonic() + 60
            return dict(state)


class DeepSeekClient:
    def __init__(self, *, base_url: str, model: str, api_key: str,
                 request_timeout_seconds: float, busy_wait_seconds: float,
                 max_tokens: int = 800, transport: httpx.AsyncBaseTransport | None = None):
        self._model, self._max_tokens = model, max_tokens
        self._request_timeout_seconds = request_timeout_seconds
        self._busy_wait_seconds = busy_wait_seconds
        self._semaphore = asyncio.Semaphore(1)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(request_timeout_seconds, connect=5),
            transport=transport,
        )
        self._probe = DeepSeekProbe(self._client, base_url, api_key, model)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        state = await self._probe.snapshot()
        return state["model_available"] is True and state["model_authenticated"] is True

    async def chat(
        self, history: Sequence[dict[str, str]], prompt: str,
        images: Sequence[EncodedImage] = (), *, identity: MemberIdentity,
        profile_targets: Sequence[MemberProfileEvidence] = (),
        system_prompt: str = SYSTEM_PROMPT, forwarded_context: str | None = None,
        wait_when_busy: bool = False, request_timeout_seconds: float | None = None,
        allow_context_timeout_extension: bool = False,
    ) -> str:
        try:
            if wait_when_busy:
                await self._semaphore.acquire()
            else:
                await asyncio.wait_for(self._semaphore.acquire(), self._busy_wait_seconds)
        except asyncio.TimeoutError:
            raise ModelBusyError("Model request slot is busy") from None
        started = time.monotonic()
        try:
            if forwarded_context is not None or any(FORWARD_CONTEXT_LABEL in item["content"] for item in history):
                system_prompt = f"{system_prompt}\n\n{FORWARD_CONTEXT_RULES}"
            text = build_current_prompt(prompt, identity, profile_targets, forwarded_context)
            content: str | list[dict] = text
            if images:
                if self._probe.supports_images is False:
                    raise ModelError("Configured model does not support images")
                content = [{"type": "text", "text": text}, *[
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image.mime_type};base64,{image.data}", "detail": "auto"}}
                    for image in images
                ]]
            payload = {"model": self._model, "messages": [
                {"role": "system", "content": system_prompt}, *history,
                {"role": "user", "content": content}],
                "stream": False, "thinking": {"type": "disabled"},
                "temperature": 0.7, "max_tokens": self._max_tokens}
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(body) > MAX_REQUEST_BYTES:
                raise ModelContextError("Request exceeds the API body limit")
            timeout = request_timeout_seconds if request_timeout_seconds is not None else self._request_timeout_seconds
            try:
                # Bound the whole request as well as each transport read.
                async with asyncio.timeout(timeout):
                    response = await self._client.post(
                        "chat/completions", content=body,
                        headers={"Content-Type": "application/json"},
                        timeout=httpx.Timeout(timeout, connect=5),
                    )
                if response.status_code in (401, 403):
                    raise ModelAuthError("Model authentication failed")
                if response.status_code == 402:
                    raise ModelBalanceError("Model balance insufficient")
                if response.status_code == 429:
                    raise ModelRateLimitError("Model rate limited")
                if response.status_code == 413:
                    raise ModelContextError("Request exceeds API limits")
                if response.status_code == 400:
                    error = response.json().get("error", {})
                    if isinstance(error, dict):
                        message = error.get("message")
                        context_message = isinstance(message, str) and any(
                            marker in message.lower() for marker in (
                                "maximum context length", "context length exceeded", "context_length_exceeded"))
                        if error.get("code") in {"context_length_exceeded", "max_tokens_exceeded"} or context_message:
                            raise ModelContextError("Input exceeds model context")
                response.raise_for_status()
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                if not isinstance(answer, str) or not answer.strip():
                    raise ModelError("Model returned an empty response")
                usage = result.get("usage")
                if isinstance(usage, dict):
                    counts = {key: value for key, value in usage.items()
                              if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                              and type(value) is int and value >= 0}
                    logger.info("model_usage provider=deepseek model=%s elapsed_ms=%d %s",
                                self._model, int((time.monotonic() - started) * 1000),
                                " ".join(f"{key}={value}" for key, value in counts.items()))
                return answer.strip()
            except (httpx.TimeoutException, TimeoutError):
                raise ModelTimeoutError("Model request timed out") from None
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, AttributeError):
                raise ModelError("Model request failed") from None
        finally:
            self._semaphore.release()
