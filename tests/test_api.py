from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from invoice_parser_llm import InvoiceParser, Settings
from invoice_parser_llm.api import create_app, resolve_parser

from .conftest import EXAMPLES, EXPECTED, FakeLLM


@pytest.fixture
def fake(acme_expected: dict[str, Any]) -> FakeLLM:
    return FakeLLM(acme_expected)


@pytest.fixture
def client(fake: FakeLLM, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("INVOICE_API_TOKEN", raising=False)
    app = create_app()
    app.dependency_overrides[resolve_parser] = lambda: InvoiceParser(client=fake, settings=Settings(model="x"))
    with TestClient(app) as c:
        yield c


def upload(client: TestClient, name: str = "acme-cloud-invoice.pdf", **kwargs: Any) -> Any:
    data = (EXAMPLES / name).read_bytes()
    return client.post("/v1/parse", files={"file": (name, data, "application/pdf")}, **kwargs)


def test_healthz_and_schema(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    assert "line_items" in client.get("/v1/schema").json()["properties"]


def test_parse_upload(client: TestClient) -> None:
    response = upload(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["file"] == "acme-cloud-invoice.pdf"
    assert body["invoice"]["total"] == 1853.4
    assert body["confidence"] == 1.0


def test_unsupported_and_empty_uploads(client: TestClient) -> None:
    r = client.post("/v1/parse", files={"file": ("x.docx", b"PK\x03\x04\xff\xfe", "application/octet-stream")})
    assert r.status_code == 415
    assert client.post("/v1/parse", files={"file": ("x.pdf", b"", "application/pdf")}).status_code == 400


def test_upload_size_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVOICE_MAX_UPLOAD_MB", "0.001")
    assert upload(client).status_code == 413


def test_bearer_token(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVOICE_API_TOKEN", "s3cret")
    assert upload(client).status_code == 401
    assert upload(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert upload(client, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_validate_endpoint(client: TestClient) -> None:
    import json

    data = json.loads((EXPECTED / "acme-cloud-invoice.json").read_text())
    assert client.post("/v1/validate", json=data).json() == []
    data["subtotal"] = 1000
    codes = [i["code"] for i in client.post("/v1/validate", json=data).json()]
    assert "subtotal_mismatch" in codes


def test_missing_configuration_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    from invoice_parser_llm import api

    monkeypatch.delenv("INVOICE_LLM_MODEL", raising=False)
    api.get_parser.cache_clear()
    with TestClient(create_app()) as c:
        response = upload(c)
    assert response.status_code == 503
    assert "INVOICE_LLM_MODEL" in response.json()["detail"]
