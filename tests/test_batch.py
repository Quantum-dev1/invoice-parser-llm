from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from invoice_parser_llm import InvoiceParser, LLMResponse, Settings
from invoice_parser_llm.batch import (
    find_documents,
    parse_many,
    write_items_csv,
    write_jsonl,
    write_summary_csv,
)
from invoice_parser_llm.llm import Message

from .conftest import EXAMPLES, load_expected


class RoutingLLM:
    """Answers with the ground truth of whichever sample invoice appears in the prompt."""

    model = "routing-fake"

    def __init__(self) -> None:
        self.truth = {
            "INV-2026-0042": load_expected("acme-cloud-invoice"),
            "R-58213": load_expected("northwind-receipt"),
            "2026-1187": load_expected("initech-office-supplies"),
        }

    def complete(self, messages: list[Message]) -> LLMResponse:
        content = messages[1]["content"]
        if isinstance(content, list):  # vision request: only the Initech scan is an image in this folder
            return LLMResponse(json.dumps(self.truth["2026-1187"]), self.model)
        for key, data in self.truth.items():
            if key in content:
                return LLMResponse(json.dumps(data), self.model)
        raise AssertionError("unknown document")


def sample_folder(tmp_path: Path) -> Path:
    for name in ("acme-cloud-invoice.pdf", "northwind-receipt.pdf", "initech-office-supplies-photo.jpg"):
        shutil.copy(EXAMPLES / name, tmp_path / name)
    (tmp_path / "notes.md").write_text("ignored")
    (tmp_path / "corrupt.pdf").write_bytes(b"%PDF-1.4 not really")
    return tmp_path


def run(tmp_path: Path) -> list[Any]:
    parser = InvoiceParser(client=RoutingLLM(), settings=Settings(model="x"))
    paths = find_documents(sample_folder(tmp_path))
    return sorted(parse_many(parser, paths, workers=3), key=lambda i: i.path.name)


def test_find_documents_filters_by_extension(tmp_path: Path) -> None:
    names = [p.name for p in find_documents(sample_folder(tmp_path))]
    assert names == [
        "acme-cloud-invoice.pdf",
        "corrupt.pdf",
        "initech-office-supplies-photo.jpg",
        "northwind-receipt.pdf",
    ]


def test_batch_isolates_failures_and_writes_jsonl(tmp_path: Path) -> None:
    items = run(tmp_path)
    assert [i.ok for i in items] == [True, False, True, True]
    assert "UnsupportedDocumentError" in (items[1].error or "")

    buf = io.StringIO()
    assert write_jsonl(items, buf) == 4
    records = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert records[0]["invoice"]["total"] == 1853.4
    assert records[0]["confidence"] == 1.0
    assert records[2]["extraction"] == "vision"
    assert "error" in records[1]


def test_summary_and_items_csv(tmp_path: Path) -> None:
    items = run(tmp_path)

    summary = io.StringIO()
    write_summary_csv(items, summary)
    rows = list(csv.DictReader(io.StringIO(summary.getvalue())))
    assert [r["status"] for r in rows] == ["ok", "error", "ok", "ok"]
    initech = rows[2]
    assert initech["vendor_name"] == "Initech Office Supplies B.V."
    assert initech["total"] == "2757.35"
    assert initech["tax_total"] == "466.65"
    assert initech["line_items"] == "4"

    lines = io.StringIO()
    assert write_items_csv(items, lines) == 3 + 4 + 3
    line_rows = list(csv.DictReader(io.StringIO(lines.getvalue())))
    assert line_rows[0]["description"].startswith("Cloud hosting")
    assert line_rows[0]["currency"] == "GBP"
