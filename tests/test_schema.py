from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoice_parser_llm import Invoice, parse_decimal
from invoice_parser_llm.prompts import SCHEMA_SKELETON


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (12, Decimal(12)),
        (12.5, Decimal("12.5")),
        ("1,234.56", Decimal("1234.56")),
        ("1.234,56", Decimal("1234.56")),
        ("€ 1.234,56", Decimal("1234.56")),
        ("$1,250", Decimal("1250")),
        ("12,50", Decimal("12.50")),
        ("1.250.000", Decimal("1250000")),
        ("(99.30)", Decimal("-99.30")),
        ("-99,30 EUR", Decimal("-99.30")),
        ("12.00-", Decimal("-12.00")),
        ("2 757,35", Decimal("2757.35")),
        ("", None),
        ("n/a", None),
        (None, None),
    ],
)
def test_parse_decimal(raw: object, expected: Decimal | None) -> None:
    assert parse_decimal(raw) == expected


@pytest.mark.parametrize("raw", ["abc", True, [1]])
def test_parse_decimal_rejects_garbage(raw: object) -> None:
    with pytest.raises(ValueError, match=r"."):
        parse_decimal(raw)


def test_invoice_normalizes_llm_output() -> None:
    inv = Invoice.model_validate(
        {
            "document_type": "Tax Invoice",
            "currency": "€",
            "issue_date": "March 3, 2026",
            "due_date": "2026-04-02T00:00:00",
            "vendor": {"name": "  Example Vendor  "},
            "line_items": [{"description": "Widget", "quantity": "2", "unit_price": "1.234,50", "amount": "2.469,00"}],
            "discount_total": -10,
            "total": "2,459.00",
            "unknown_field": "ignored",
        }
    )
    assert inv.document_type == "invoice"
    assert inv.currency == "EUR"
    assert inv.issue_date == date(2026, 3, 3)
    assert inv.due_date == date(2026, 4, 2)
    assert inv.vendor.name == "Example Vendor"
    assert inv.line_items[0].unit_price == Decimal("1234.50")
    assert inv.discount_total == Decimal(10)
    assert inv.total == Decimal("2459.00")


def test_unknown_document_type_maps_to_other() -> None:
    assert Invoice.model_validate({"document_type": "delivery note"}).document_type == "other"


def test_rejects_ambiguous_numeric_dates_and_bad_currency() -> None:
    with pytest.raises(ValidationError, match="ambiguous"):
        Invoice.model_validate({"issue_date": "03/04/2026"})
    with pytest.raises(ValidationError, match="ISO 4217"):
        Invoice.model_validate({"currency": "dollars"})


def test_money_serializes_as_json_numbers() -> None:
    inv = Invoice.model_validate({"total": "1,853.40", "subtotal": "1544.00"})
    data = json.loads(inv.model_dump_json())
    assert data["total"] == 1853.4
    assert data["subtotal"] == 1544
    assert isinstance(inv.total, Decimal)


def test_prompt_skeleton_matches_schema() -> None:
    """The hand-written prompt skeleton must list exactly the model's fields."""
    assert set(SCHEMA_SKELETON) == set(Invoice.model_fields)
    from invoice_parser_llm.schema import LineItem, Party, TaxLine

    assert set(SCHEMA_SKELETON["vendor"]) == set(Party.model_fields)
    assert set(SCHEMA_SKELETON["line_items"][0]) == set(LineItem.model_fields)
    assert set(SCHEMA_SKELETON["taxes"][0]) == set(TaxLine.model_fields)
