"""End-to-end over real HTTP: the real OpenAICompatibleClient talks to a local stub server that implements
the OpenAI chat-completions protocol. No external network, no API keys."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from invoice_parser_llm import InvoiceParser, Settings

from .conftest import EXAMPLES, load_expected


class StubOpenAI(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    truth = {
        "INV-2026-0042": load_expected("acme-cloud-invoice"),
        "2026-1187": load_expected("initech-office-supplies"),
    }

    def log_message(self, *args: Any) -> None:  # silence
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubOpenAI.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        user = body["messages"][1]["content"]
        if isinstance(user, list):
            answer = self.truth["2026-1187"]
        else:
            answer = next(v for k, v in self.truth.items() if k in user)
        # Wrap in a code fence like many local models do.
        payload = {
            "model": body["model"],
            "choices": [{"message": {"role": "assistant", "content": f"```json\n{json.dumps(answer)}\n```"}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture(scope="module")
def stub_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_text_pdf_over_http(stub_url: str) -> None:
    StubOpenAI.requests.clear()
    parser = InvoiceParser(settings=Settings(model="local-model", base_url=stub_url, api_key="test-key"))
    result = parser.parse(EXAMPLES / "acme-cloud-invoice.pdf")

    assert result.invoice is not None and result.needs_review is False
    assert result.model == "local-model"
    assert result.usage == {"prompt_tokens": 42, "completion_tokens": 7}
    req = StubOpenAI.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["auth"] == "Bearer test-key"
    assert req["body"]["response_format"] == {"type": "json_object"}


def test_vision_model_over_http(stub_url: str) -> None:
    StubOpenAI.requests.clear()
    settings = Settings(model="text-model", vision_model="vision-model", base_url=stub_url)
    result = InvoiceParser(settings=settings).parse(EXAMPLES / "initech-office-supplies-photo.jpg")

    assert result.extraction == "vision"
    assert result.invoice is not None and str(result.invoice.total) == "2757.35"
    assert StubOpenAI.requests[0]["body"]["model"] == "vision-model"
    assert StubOpenAI.requests[0]["auth"] is None
