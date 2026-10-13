from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from invoice_parser_llm import InvoiceParser, Settings, cli

from .conftest import EXAMPLES, EXPECTED, FakeLLM
from .test_batch import RoutingLLM

runner = CliRunner()


@pytest.fixture
def fake_parser(monkeypatch: pytest.MonkeyPatch, acme_expected: dict[str, Any]) -> FakeLLM:
    fake = FakeLLM(acme_expected)
    monkeypatch.setattr(cli, "make_parser", lambda *a, **k: InvoiceParser(client=fake, settings=Settings(model="x")))
    return fake


def test_parse_prints_json(fake_parser: FakeLLM) -> None:
    result = runner.invoke(cli.app, ["parse", str(EXAMPLES / "acme-cloud-invoice.pdf"), "--quiet"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["invoice"]["invoice_number"] == "INV-2026-0042"
    assert data["needs_review"] is False


def test_parse_invoice_only_to_file(fake_parser: FakeLLM, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    result = runner.invoke(
        cli.app, ["parse", str(EXAMPLES / "acme-cloud-invoice.pdf"), "--invoice-only", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["total"] == 1853.4


def test_parse_strict_exit_code(monkeypatch: pytest.MonkeyPatch, acme_expected: dict[str, Any]) -> None:
    wrong = {**acme_expected, "total": 1.0, "amount_due": None}
    fake = FakeLLM(wrong, wrong, wrong)
    monkeypatch.setattr(cli, "make_parser", lambda *a, **k: InvoiceParser(client=fake, settings=Settings(model="x")))
    result = runner.invoke(cli.app, ["parse", str(EXAMPLES / "acme-cloud-invoice.pdf"), "--strict"])
    assert result.exit_code == 2


def test_parse_reports_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVOICE_LLM_MODEL", raising=False)
    result = runner.invoke(cli.app, ["parse", str(EXAMPLES / "acme-cloud-invoice.pdf")])
    assert result.exit_code == 1
    assert "INVOICE_LLM_MODEL" in result.output


def test_batch_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        cli, "make_parser", lambda *a, **k: InvoiceParser(client=RoutingLLM(), settings=Settings(model="x"))
    )
    folder = tmp_path / "in"
    folder.mkdir()
    for name in ("acme-cloud-invoice.pdf", "northwind-receipt.pdf"):
        shutil.copy(EXAMPLES / name, folder / name)
    out, summary, items = tmp_path / "r.jsonl", tmp_path / "s.csv", tmp_path / "i.csv"
    result = runner.invoke(
        cli.app,
        [
            "batch",
            str(folder),
            "-o",
            str(out),
            "--csv",
            str(summary),
            "--items-csv",
            str(items),
            "--workers",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(out.read_text().splitlines()) == 2
    assert len(list(csv.DictReader(summary.open()))) == 2
    assert len(list(csv.DictReader(items.open()))) == 6
    assert "2 ok" in result.output


def test_validate_command(tmp_path: Path) -> None:
    good = runner.invoke(cli.app, ["validate", str(EXPECTED / "initech-office-supplies.json")])
    assert good.exit_code == 0
    assert json.loads(good.stdout) == []

    data = json.loads((EXPECTED / "initech-office-supplies.json").read_text())
    data["total"] = 1
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"invoice": data}))
    bad = runner.invoke(cli.app, ["validate", str(bad_file)])
    assert bad.exit_code == 2
    assert "total_mismatch" in bad.stdout


def test_schema_and_version() -> None:
    schema = runner.invoke(cli.app, ["schema"])
    assert schema.exit_code == 0
    assert "line_items" in json.loads(schema.stdout)["properties"]
    version = runner.invoke(cli.app, ["--version"])
    assert "0.1.0" in version.stdout
