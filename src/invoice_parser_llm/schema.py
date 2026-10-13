"""Pydantic v2 models for extracted invoices.

Money values are ``Decimal`` in Python (exact arithmetic for validation) and are serialized as JSON
numbers. Inputs are forgiving: ``"1,234.50"``, ``"€ 1.234,50"``, ``"(12.00)"`` and plain numbers are
all accepted, because LLMs sometimes copy amounts verbatim from the document.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, field_validator

_CURRENCY_SYMBOLS = {
    "€": "EUR",
    "£": "GBP",
    "US$": "USD",
    "$": "USD",
    "₹": "INR",
    "₩": "KRW",
    "₺": "TRY",
    "₪": "ILS",
    "₫": "VND",
    "₱": "PHP",
    "฿": "THB",
    "R$": "BRL",
    "A$": "AUD",
    "C$": "CAD",
    "CHF": "CHF",
}

_STRIP_NON_NUMERIC = re.compile(r"[^\d,.\-+]")


def parse_decimal(value: Any) -> Decimal | None:
    """Coerce numbers and human-formatted amount strings to ``Decimal``.

    Handles thousands separators in both styles (``1,234.56`` and ``1.234,56``), currency symbols,
    surrounding text, and accounting negatives in parentheses. Returns ``None`` for empty values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid amount")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    if not isinstance(value, str):
        raise ValueError(f"cannot parse amount from {type(value).__name__}")

    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "-", "—"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = _STRIP_NON_NUMERIC.sub("", text)
    if cleaned.endswith("-"):  # trailing minus, e.g. "12.00-"
        negative = True
        cleaned = cleaned[:-1]
    if not cleaned or not re.search(r"\d", cleaned):
        raise ValueError(f"no digits in amount {value!r}")

    if "," in cleaned and "." in cleaned:
        # The right-most separator is the decimal separator.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # "12,50" → decimal comma; "1,250" / "1,250,000" → thousands separators.
        head, _, tail = cleaned.rpartition(",")
        cleaned = f"{head.replace(',', '')}.{tail}" if len(tail) in (1, 2) else cleaned.replace(",", "")
    elif cleaned.count(".") > 1:
        # "1.250.000" → thousands separators.
        cleaned = cleaned.replace(".", "")

    try:
        result = Decimal(cleaned)
    except InvalidOperation as exc:
        raise ValueError(f"invalid amount {value!r}") from exc
    return -abs(result) if negative else result


def _serialize_decimal(value: Decimal) -> float | int:
    return int(value) if value == value.to_integral_value() and abs(value) < 2**53 else float(value)


Money = Annotated[
    Decimal,
    BeforeValidator(parse_decimal),
    PlainSerializer(_serialize_decimal, return_type=int | float, when_used="json"),
]
"""A monetary amount (or quantity / percentage) parsed leniently and serialized as a JSON number."""

_TEXT_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
)


def parse_date(value: Any) -> Any:
    """Accept ISO dates plus unambiguous textual formats (``March 3, 2026``, ``3 Mar 2026``).

    Purely numeric day/month orders such as ``03/04/2026`` are ambiguous and are rejected so the
    model is asked to resolve them instead of guessing silently.
    """
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"null", "none", "n/a"}:
            return None
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            pass
        for fmt in _TEXT_DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"unrecognized or ambiguous date {value!r}; use YYYY-MM-DD")
    return value


IsoDate = Annotated[date, BeforeValidator(parse_date)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True, populate_by_name=True)


class Party(_Model):
    """A vendor (seller/issuer) or buyer (customer/bill-to)."""

    name: str | None = None
    address: str | None = None
    tax_id: str | None = Field(default=None, description="VAT / GST / EIN / company tax identifier as printed")
    email: str | None = None
    phone: str | None = None


class LineItem(_Model):
    description: str
    sku: str | None = None
    quantity: Money | None = None
    unit: str | None = Field(default=None, description="Unit of measure, e.g. 'h', 'pcs', 'kg'")
    unit_price: Money | None = None
    tax_rate: Money | None = Field(default=None, description="Tax rate in percent, e.g. 20 for 20%")
    amount: Money = Field(description="Line total as printed (usually quantity × unit price, before tax)")


class TaxLine(_Model):
    name: str | None = Field(default=None, description="e.g. 'VAT', 'GST', 'Sales tax'")
    rate: Money | None = Field(default=None, description="Rate in percent, e.g. 20 for 20%")
    base: Money | None = Field(default=None, description="Taxable amount the rate applies to")
    amount: Money


DocumentType = Literal["invoice", "receipt", "credit_note", "proforma", "other"]


class Invoice(_Model):
    """Structured invoice / receipt data."""

    document_type: DocumentType = "invoice"
    invoice_number: str | None = None
    issue_date: IsoDate | None = None
    due_date: IsoDate | None = None
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. USD, EUR, GBP")
    vendor: Party = Field(default_factory=Party)
    buyer: Party | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Money | None = Field(default=None, description="Sum of line items before discounts and tax")
    discount_total: Money | None = Field(default=None, description="Total discount as a positive number")
    shipping_total: Money | None = None
    taxes: list[TaxLine] = Field(default_factory=list)
    tax_total: Money | None = None
    total: Money | None = Field(default=None, description="Grand total including tax")
    amount_paid: Money | None = None
    amount_due: Money | None = None
    payment_terms: str | None = None
    notes: str | None = None

    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, value: Any) -> Any:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        if text in _CURRENCY_SYMBOLS:
            return _CURRENCY_SYMBOLS[text]
        code = text.upper()
        if not re.fullmatch(r"[A-Z]{3}", code):
            raise ValueError(f"currency must be an ISO 4217 code like 'USD', got {value!r}")
        return code

    @field_validator("discount_total", mode="after")
    @classmethod
    def _discount_positive(cls, value: Decimal | None) -> Decimal | None:
        return abs(value) if value is not None else None

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            v = value.strip().lower().replace(" ", "_").replace("-", "_")
            aliases = {
                "tax_invoice": "invoice",
                "bill": "invoice",
                "sales_receipt": "receipt",
                "credit_memo": "credit_note",
            }
            v = aliases.get(v, v)
            return v if v in {"invoice", "receipt", "credit_note", "proforma", "other"} else "other"
        return value


def invoice_json_schema() -> dict[str, Any]:
    """JSON Schema of :class:`Invoice` (used in prompts and exposed by the API/CLI)."""
    return Invoice.model_json_schema()
