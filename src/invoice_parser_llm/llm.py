"""Minimal client for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with OpenAI, OpenRouter, Ollama (``/v1``), vLLM, LM Studio, llama.cpp
server and others. Uses plain ``httpx`` — no vendor SDK.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

Message = dict[str, Any]

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """The LLM endpoint failed or returned an unusable response."""


@dataclass
class LLMResponse:
    content: str
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that can answer a chat completion. Implement this to plug in another backend or a mock."""

    model: str

    def complete(self, messages: list[Message]) -> LLMResponse: ...


class OpenAICompatibleClient:
    """Synchronous chat-completions client with retries and JSON-mode auto-fallback."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        json_mode: bool = True,
        temperature: float = 0.0,
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.temperature = temperature
        self._sleep = sleep
        headers = {"Content-Type": "application/json", **(extra_headers or {})}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(timeout=timeout, headers=headers, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _payload(self, messages: list[Message]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, messages: list[Message]) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        attempt = 0
        while True:
            try:
                response = self._http.post(url, json=self._payload(messages))
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"Could not reach {url}: {exc}") from exc
                self._backoff(attempt, None)
                attempt += 1
                continue

            if response.status_code == 400 and self.json_mode and "response_format" in response.text:
                # Some servers do not support JSON mode; the prompt still asks for JSON, so just drop it.
                logger.info("Endpoint rejected response_format; retrying without JSON mode")
                self.json_mode = False
                continue

            if response.status_code in _RETRY_STATUS and attempt < self.max_retries:
                self._backoff(attempt, response.headers.get("retry-after"))
                attempt += 1
                continue

            if response.status_code >= 400:
                raise LLMError(f"LLM endpoint returned HTTP {response.status_code}: {response.text[:500]}")

            try:
                body = response.json()
                content = body["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMError(f"Unexpected response shape from {url}: {response.text[:500]}") from exc
            if isinstance(content, list):  # some servers return content parts
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            if not isinstance(content, str) or not content.strip():
                raise LLMError("LLM returned an empty message")
            return LLMResponse(content=content, model=body.get("model"), usage=body.get("usage") or {})

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay: float
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        if delay <= 0:
            delay = min(2.0**attempt, 20.0) + random.uniform(0, 0.25)
        self._sleep(min(delay, 60.0))
