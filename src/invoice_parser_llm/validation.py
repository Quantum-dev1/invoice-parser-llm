"""Deterministic arithmetic and consistency checks for extracted invoices.

LLMs are good at reading documents and bad at noticing that they misread a digit. These checks catch
most extraction errors cheaply: if the numbers do not add up, something was read wrong (or the
document itself is wrong — either way a human should look).
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from .schema import Invoice, Money

Severity = Literal["error", "warning"]

DEFAULT_TOLERANCE = Decimal("0.01")
_HUNDRED = Decimal(100)


class ValidationIssue(BaseModel):
    """One problem found in an extracted invoice."""

    code: str
    severity: Severity
    message: str
    field: str | None = None
    expected: Money | None = None
    actual: Money | None = None


def _sum(values: Iterable[Decimal]) -> Decimal:
    return sum(values, Decimal(0))


def _fmt(value: Decimal) -> str:
    return f"{value:,.2f}"


class _Checker:
    def __init__(self, tolerance: Decimal) -> None:
        self.tolerance = tolerance
        self.issues: list[ValidationIssue] = []

    def close(self, expected: Decimal, actual: Decimal, terms: int = 1) -> bool:
        # Rounding errors accumulate per summed term, so the allowed difference scales with the term count.
        allowed = self.tolerance * max(terms, 1)
        return abs(expected - actual) <= allowed

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        field: str | None = None,
        expected: Decimal | None = None,
        actual: Decimal | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code, severity=severity, message=message, field=field, expected=expected, actual=actual
            )
        )


def validate_invoice(invoice: Invoice, tolerance: Decimal | float | str = DEFAULT_TOLERANCE) -> list[ValidationIssue]:
    """Run all checks and return the issues found (empty list = consistent).

    ``tolerance`` is the allowed absolute difference per summed term (default 0.01), so a sum of
    ten rounded line items may be off by up to 0.10.
    """
    c = _Checker(Decimal(str(tolerance)))
    inv = invoice

    # -- required-ish fields ---------------------------------------------------------------------------
    if inv.total is None:
        c.add("missing_total", "error", "No grand total was extracted.", "total")
    if not inv.vendor.name:
        c.add("missing_vendor", "warning", "Vendor name is missing.", "vendor.name")
    if not inv.currency:
        c.add("missing_currency", "warning", "Currency is missing.", "currency")
    if inv.issue_date is None:
        c.add("missing_issue_date", "warning", "Issue date is missing.", "issue_date")
    if not inv.invoice_number and inv.document_type in ("invoice", "credit_note", "proforma"):
        c.add("missing_invoice_number", "warning", "Invoice number is missing.", "invoice_number")

    # -- line items -----------------------------------------------------------------------------------
    for i, item in enumerate(inv.line_items):
        if item.quantity is not None and item.unit_price is not None:
            expected = item.quantity * item.unit_price
            if not c.close(expected, item.amount):
                c.add(
                    "line_amount_mismatch",
                    "error",
                    f"Line {i + 1} ({item.description!r}): {item.quantity} × {_fmt(item.unit_price)} = "
                    f"{_fmt(expected)}, but amount is {_fmt(item.amount)}.",
                    f"line_items[{i}].amount",
                    expected,
                    item.amount,
                )

    lines_sum = _sum(item.amount for item in inv.line_items) if inv.line_items else None
    n_lines = len(inv.line_items)

    tax_sum = _sum(t.amount for t in inv.taxes) if inv.taxes else None
    tax = inv.tax_total if inv.tax_total is not None else tax_sum

    if lines_sum is not None and inv.subtotal is not None and not c.close(inv.subtotal, lines_sum, n_lines):
        if tax is not None and c.close(inv.subtotal + tax, lines_sum, n_lines):
            c.add(
                "line_items_tax_inclusive",
                "warning",
                "Line items appear to include tax (they sum to subtotal + tax).",
                "line_items",
                inv.subtotal,
                lines_sum,
            )
        else:
            c.add(
                "subtotal_mismatch",
                "error",
                f"Line items sum to {_fmt(lines_sum)}, but subtotal is {_fmt(inv.subtotal)}.",
                "subtotal",
                lines_sum,
                inv.subtotal,
            )

    # -- taxes ----------------------------------------------------------------------------------------
    if tax_sum is not None and inv.tax_total is not None and not c.close(inv.tax_total, tax_sum, len(inv.taxes)):
        c.add(
            "tax_total_mismatch",
            "error",
            f"Tax lines sum to {_fmt(tax_sum)}, but tax total is {_fmt(inv.tax_total)}.",
            "tax_total",
            tax_sum,
            inv.tax_total,
        )
    for i, t in enumerate(inv.taxes):
        if t.rate is not None and t.base is not None:
            expected = (t.base * t.rate / _HUNDRED).quantize(Decimal("0.01"))
            if not c.close(expected, t.amount):
                c.add(
                    "tax_rate_mismatch",
                    "warning",
                    f"Tax line {i + 1}: {t.rate}% of {_fmt(t.base)} is {_fmt(expected)}, "
                    f"but amount is {_fmt(t.amount)}.",
                    f"taxes[{i}].amount",
                    expected,
                    t.amount,
                )

    # -- grand total ----------------------------------------------------------------------------------
    base = inv.subtotal if inv.subtotal is not None else lines_sum
    if inv.total is not None and base is not None:
        expected_total = base - (inv.discount_total or 0) + (inv.shipping_total or 0) + (tax or 0)
        terms = 1 + (n_lines if inv.subtotal is None else 0) + len(inv.taxes)
        if not c.close(expected_total, inv.total, terms):
            already_explained = any(i.code == "line_items_tax_inclusive" for i in c.issues) and c.close(
                base - (inv.discount_total or 0) + (inv.shipping_total or 0), inv.total, terms
            )
            if not already_explained:
                parts = [f"subtotal {_fmt(base)}"]
                if inv.discount_total:
                    parts.append(f"− discount {_fmt(inv.discount_total)}")
                if inv.shipping_total:
                    parts.append(f"+ shipping {_fmt(inv.shipping_total)}")
                if tax:
                    parts.append(f"+ tax {_fmt(tax)}")
                c.add(
                    "total_mismatch",
                    "error",
                    f"{' '.join(parts)} = {_fmt(expected_total)}, but total is {_fmt(inv.total)}.",
                    "total",
                    expected_total,
                    inv.total,
                )

    # -- payment --------------------------------------------------------------------------------------
    if inv.total is not None and inv.amount_due is not None:
        expected_due = inv.total - (inv.amount_paid or 0)
        if not c.close(expected_due, inv.amount_due):
            c.add(
                "amount_due_mismatch",
                "warning",
                f"Total {_fmt(inv.total)} − paid {_fmt(inv.amount_paid or Decimal(0))} = {_fmt(expected_due)}, "
                f"but amount due is {_fmt(inv.amount_due)}.",
                "amount_due",
                expected_due,
                inv.amount_due,
            )

    # -- sanity ---------------------------------------------------------------------------------------
    if inv.issue_date and inv.due_date and inv.due_date < inv.issue_date:
        c.add("due_before_issue", "warning", "Due date is before the issue date.", "due_date")
    if inv.total is not None and inv.total < 0 and inv.document_type != "credit_note":
        c.add("negative_total", "warning", "Total is negative but the document is not a credit note.", "total")

    return c.issues


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
