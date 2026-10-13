from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from invoice_parser_llm import Invoice, InvoiceParser, LLMResponse, Settings, cli
from invoice_parser_llm.evaluation import FIELD_CHECKS, compare, match_truth
from invoice_parser_llm.llm import Message

from .conftest import EXAMPLES, EXPECTED
from .test_batch import RoutingLLM


def test_compare_identical_is_perfect(initech_expected: dict[str, Any]) -> None:
    inv = Invoice.model_validate(initech_expected)
    assert all(compare(inv, inv).values())
    assert not any(compare(None, inv).values())


def test_compare_is_lenient_on_formatting_only(acme_expected: dict[str, Any]) -> None:
    truth = Invoice.model_validate(acme_expected)
    pred_data = copy.deepcopy(acme_expected)
    pred_data["vendor"]["name"] = "ACME CLOUD SERVICES LTD"  # case / trailing dot
    pred_data["vendor"]["tax_id"] = "GB 000 000 042"  # spacing
    pred_data["line_items"].reverse()  # order does not matter for amounts
    pred_data["total"] = 1853.41  # wrong by a cent+ → counted as wrong
    result = compare(Invoice.model_validate(pred_data), truth)
    assert result["vendor.name"] and result["vendor.tax_id"] and result["line_items.amounts"]
    assert result["total"] is False


def test_match_truth_by_stem_and_variant() -> None:
    truths = sorted(EXPECTED.glob("*.json"))
    assert match_truth(Path("initech-office-supplies-photo.jpg"), truths) == EXPECTED / "initech-office-supplies.json"
    assert match_truth(Path("acme-cloud-invoice.pdf"), truths) == EXPECTED / "acme-cloud-invoice.json"
    assert match_truth(Path("unrelated.pdf"), truths) is None


class OneDigitOff(RoutingLLM):
    """Gets the Northwind subtotal wrong every time (even after correction requests)."""

    def complete(self, messages: list[Message]) -> LLMResponse:
        response = super().complete(messages)
        data = json.loads(response.content)
        if data.get("invoice_number") == "R-58213":
            data["subtotal"] = 13.45
        return LLMResponse(json.dumps(data), self.model)


@pytest.mark.parametrize(("llm", "expected_overall"), [(RoutingLLM, 1.0), (OneDigitOff, None)])
def test_eval_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, llm: type, expected_overall: float | None
) -> None:
    monkeypatch.setattr(cli, "make_parser", lambda *a, **k: InvoiceParser(client=llm(), settings=Settings(model="x")))
    report_path = tmp_path / "report.json"
    result = CliRunner().invoke(cli.app, ["eval", str(EXAMPLES), str(EXPECTED), "--report", str(report_path)])
    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    assert len(report["documents"]) == 5  # 3 digital PDFs + photo + scanned PDF
    if expected_overall is not None:
        assert report["overall_accuracy"] == expected_overall
        assert report["exact_documents"] == 5
    else:
        assert report["field_accuracy"]["subtotal"] == 0.8
        northwind = next(d for d in report["documents"] if d["file"] == "northwind-receipt.pdf")
        assert northwind["fields"]["subtotal"] is False
        assert northwind["needs_review"] is True
    assert set(report["field_accuracy"]) == set(FIELD_CHECKS)
