from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from invoice_parser_llm import LLMError, OpenAICompatibleClient


def ok(content: str = '{"total": 1}') -> httpx.Response:
    return httpx.Response(
        200,
        json={"model": "m-1", "choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 3}},
    )


def client_with(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: object
) -> tuple[OpenAICompatibleClient, list[float]]:
    sleeps: list[float] = []
    client = OpenAICompatibleClient(
        model="m",
        base_url="http://llm.local/v1/",
        api_key=kwargs.pop("api_key", "secret"),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
        sleep=sleeps.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return client, sleeps


def test_request_shape_and_response_parsing() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok()

    client, _ = client_with(handler)
    response = client.complete([{"role": "user", "content": "hi"}])

    assert response.content == '{"total": 1}'
    assert response.model == "m-1"
    assert response.usage == {"prompt_tokens": 3}
    req = seen[0]
    assert str(req.url) == "http://llm.local/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer secret"
    body = json.loads(req.content)
    assert body == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def test_no_auth_header_without_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok()

    client, _ = client_with(handler, api_key=None)
    client.complete([])
    assert "authorization" not in seen[0].headers


def test_retries_on_rate_limit_and_server_errors() -> None:
    responses = iter([httpx.Response(429, headers={"retry-after": "3"}), httpx.Response(503), ok()])
    client, sleeps = client_with(lambda _req: next(responses))
    assert client.complete([]).content == '{"total": 1}'
    assert sleeps[0] == 3.0
    assert len(sleeps) == 2


def test_retries_on_transport_errors_then_gives_up() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client, sleeps = client_with(handler, max_retries=2)
    with pytest.raises(LLMError, match="Could not reach"):
        client.complete([])
    assert len(sleeps) == 2


def test_falls_back_when_json_mode_is_unsupported() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return ok()

    client, _ = client_with(handler)
    assert client.complete([]).content
    assert "response_format" in bodies[0] and "response_format" not in bodies[1]
    client.complete([])
    assert len(bodies) == 3  # remembered: no extra failing request


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(401, text="bad key"), "HTTP 401"),
        (httpx.Response(200, text="<html>"), "Unexpected response"),
        (httpx.Response(200, json={"choices": []}), "Unexpected response"),
        (httpx.Response(200, json={"choices": [{"message": {"content": "  "}}]}), "empty"),
    ],
)
def test_errors(response: httpx.Response, match: str) -> None:
    client, _ = client_with(lambda _req: response)
    with pytest.raises(LLMError, match=match):
        client.complete([])


def test_content_parts_are_joined() -> None:
    client, _ = client_with(
        lambda _req: httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "{}"}]}}]})
    )
    assert client.complete([]).content == "{}"
