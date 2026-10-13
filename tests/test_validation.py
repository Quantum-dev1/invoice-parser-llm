from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import pytest

from invoice_parser_llm import Invoice, validate_invoice


def codes(data: dict[str, Any], tolerance: str = "0.01") -> list[str]:
    return [i.code for i in validate_invoice(Invoice.model_validate(data), tolerance)]


@pytest.mark.parametrize("name", ["acme_expected", "initech_expected", "northwind_expected"])
def test_ground_truth_samples_are_consistent(name: str, request: pytest.FixtureRequest) -> None:
    assert codes(request.getfixturevalue(name)) == []


def test_line_amount_mismatch(acme_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(acme_expected)
    data["line_items"][2]["amount"] = 671.50  # misread 617.50
    result = codes(data)
    assert "line_amount_mismatch" in result
    assert "subtotal_mismatch" in result


def test_missed_line_item_is_caught(initech_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(initech_expected)
    del data["line_items"][3]
    assert codes(data) == ["subtotal_mismatch"]


def test_total_mismatch_with_discount_and_shipping(initech_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(initech_expected)
    data["total"] = 2775.35  # digit swap
    issues = validate_invoice(Invoice.model_validate(data))
    assert [i.code for i in issues] == ["total_mismatch", "amount_due_mismatch"]
    total_issue = issues[0]
    assert total_issue.expected == Decimal("2757.35")
    assert "discount" in total_issue.message and "shipping" in total_issue.message


def test_tax_lines_must_sum_to_tax_total(initech_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(initech_expected)
    data["tax_total"] = 455.85  # only one of two VAT lines
    assert "tax_total_mismatch" in codes(data)


def test_tax_rate_mismatch_is_a_warning(acme_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(acme_expected)
    data["taxes"][0]["rate"] = 21
    issues = validate_invoice(Invoice.model_validate(data))
    assert [(i.code, i.severity) for i in issues] == [("tax_rate_mismatch", "warning")]


def test_tax_inclusive_line_items_are_a_warning_not_an_error() -> None:
    data = {
        "invoice_number": "1",
        "issue_date": "2026-01-01",
        "currency": "GBP",
        "vendor": {"name": "Example Shop"},
        "line_items": [{"description": "Thing", "amount": 120}],
        "subtotal": 100,
        "tax_total": 20,
        "total": 120,
    }
    assert codes(data) == ["line_items_tax_inclusive"]


def test_tolerance_scales_with_number_of_lines() -> None:
    lines = [{"description": f"x{i}", "quantity": 1, "unit_price": "0.333", "amount": "0.33"} for i in range(3)]
    base = {"invoice_number": "1", "issue_date": "2026-01-01", "currency": "USD", "vendor": {"name": "V"}}
    ok = {**base, "line_items": lines, "subtotal": "1.00", "total": "1.00"}
    assert codes(ok) == []  # 0.99 vs 1.00 is within 3 × 0.01
    assert "subtotal_mismatch" in codes({**ok, "subtotal": "1.10", "total": "1.10"})


def test_missing_fields_and_sanity_warnings() -> None:
    result = codes({"due_date": "2026-01-01", "issue_date": "2026-02-01", "total": -5})
    assert result == [
        "missing_vendor",
        "missing_currency",
        "missing_invoice_number",
        "due_before_issue",
        "negative_total",
    ]
    assert codes({})[0] == "missing_total"


def test_receipts_do_not_require_invoice_number(northwind_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(northwind_expected)
    data["invoice_number"] = None
    assert codes(data) == []


def test_amount_due_mismatch(northwind_expected: dict[str, Any]) -> None:
    data = copy.deepcopy(northwind_expected)
    data["amount_due"] = 34.04
    assert codes(data) == ["amount_due_mismatch"]
