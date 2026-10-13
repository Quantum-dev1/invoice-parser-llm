"""Field-level accuracy against labeled ground truth — compare models, prompts and endpoints on your own invoices."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schema import Invoice

_CENT = Decimal("0.005")


def _norm_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip().rstrip(".").casefold()
    return text or None


def _same_amount(a: Decimal | None, b: Decimal | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= _CENT


def _tax(inv: Invoice) -> Decimal | None:
    if inv.tax_total is not None:
        return inv.tax_total
    if inv.taxes:
        return sum((t.amount for t in inv.taxes), Decimal(0))
    return None


def _line_amounts(inv: Invoice) -> list[Decimal]:
    return sorted(item.amount for item in inv.line_items)


def _text(getter: Callable[[Invoice], str | None]) -> Callable[[Invoice, Invoice], bool]:
    return lambda p, t: _norm_text(getter(p)) == _norm_text(getter(t))


def _amount(getter: Callable[[Invoice], Decimal | None]) -> Callable[[Invoice, Invoice], bool]:
    return lambda p, t: _same_amount(getter(p), getter(t))


FIELD_CHECKS: dict[str, Callable[[Invoice, Invoice], bool]] = {
    "document_type": lambda p, t: p.document_type == t.document_type,
    "invoice_number": _text(lambda i: i.invoice_number),
    "issue_date": lambda p, t: p.issue_date == t.issue_date,
    "due_date": lambda p, t: p.due_date == t.due_date,
    "currency": lambda p, t: p.currency == t.currency,
    "vendor.name": _text(lambda i: i.vendor.name),
    "vendor.tax_id": lambda p, t: (p.vendor.tax_id or "").replace(" ", "") == (t.vendor.tax_id or "").replace(" ", ""),
    "buyer.name": _text(lambda i: i.buyer.name if i.buyer else None),
    "subtotal": _amount(lambda i: i.subtotal),
    "discount_total": _amount(lambda i: i.discount_total),
    "shipping_total": _amount(lambda i: i.shipping_total),
    "tax_total": _amount(_tax),
    "total": _amount(lambda i: i.total),
    "amount_due": _amount(lambda i: i.amount_due),
    "line_items.count": lambda p, t: len(p.line_items) == len(t.line_items),
    "line_items.amounts": lambda p, t: (
        len(p.line_items) == len(t.line_items)
        and all(_same_amount(a, b) for a, b in zip(_line_amounts(p), _line_amounts(t), strict=True))
    ),
}


def compare(predicted: Invoice | None, truth: Invoice) -> dict[str, bool]:
    """Per-field correctness of one prediction. A missing prediction scores every field as wrong."""
    if predicted is None:
        return dict.fromkeys(FIELD_CHECKS, False)
    return {name: check(predicted, truth) for name, check in FIELD_CHECKS.items()}


class DocumentScore(BaseModel):
    file: str
    truth: str
    fields: dict[str, bool]
    accuracy: float
    needs_review: bool | None = None


class EvalReport(BaseModel):
    documents: list[DocumentScore]
    field_accuracy: dict[str, float]
    overall_accuracy: float
    exact_documents: int


def match_truth(document: Path, truth_files: Iterable[Path]) -> Path | None:
    """Find ground truth by file stem; variants like ``foo-photo.jpg`` match ``foo.json`` (longest prefix wins)."""
    stem = document.stem
    best: Path | None = None
    for candidate in truth_files:
        if (stem == candidate.stem or stem.startswith(candidate.stem + "-")) and (
            best is None or len(candidate.stem) > len(best.stem)
        ):
            best = candidate
    return best


def load_truth(path: Path) -> Invoice:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("invoice"), dict):
        data = data["invoice"]
    return Invoice.model_validate(data)


def build_report(scores: list[DocumentScore]) -> EvalReport:
    field_accuracy = {
        name: round(sum(s.fields[name] for s in scores) / len(scores), 3) if scores else 0.0 for name in FIELD_CHECKS
    }
    total_checks = sum(len(s.fields) for s in scores)
    overall = round(sum(sum(s.fields.values()) for s in scores) / total_checks, 3) if total_checks else 0.0
    exact = sum(1 for s in scores if all(s.fields.values()))
    return EvalReport(documents=scores, field_accuracy=field_accuracy, overall_accuracy=overall, exact_documents=exact)


def score_document(file: str, truth_path: Path, predicted: Invoice | None, needs_review: bool | None) -> DocumentScore:
    fields = compare(predicted, load_truth(truth_path))
    return DocumentScore(
        file=file,
        truth=truth_path.name,
        fields=fields,
        accuracy=round(sum(fields.values()) / len(fields), 3),
        needs_review=needs_review,
    )
