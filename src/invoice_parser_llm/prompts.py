"""Prompts. Kept in one place so they are easy to audit and tune."""

from __future__ import annotations

import base64
import json
from typing import Any

from .documents import Document
from .llm import Message
from .validation import ValidationIssue

# A compact, LLM-friendly skeleton of the Invoice schema. tests/test_prompts.py keeps it in sync
# with the Pydantic model so the prompt can never drift from the schema.
SCHEMA_SKELETON: dict[str, Any] = {
    "document_type": "invoice | receipt | credit_note | proforma | other",
    "invoice_number": "string or null",
    "issue_date": "YYYY-MM-DD or null",
    "due_date": "YYYY-MM-DD or null",
    "currency": "ISO 4217 code, e.g. USD / EUR / GBP, or null",
    "vendor": {"name": "", "address": "", "tax_id": "", "email": "", "phone": ""},
    "buyer": {"name": "", "address": "", "tax_id": "", "email": "", "phone": ""},
    "line_items": [
        {
            "description": "",
            "sku": None,
            "quantity": 0,
            "unit": None,
            "unit_price": 0,
            "tax_rate": "percent, e.g. 20",
            "amount": "line total as printed",
        }
    ],
    "subtotal": "number or null",
    "discount_total": "positive number or null",
    "shipping_total": "number or null",
    "taxes": [{"name": "VAT", "rate": 20, "base": 0, "amount": 0}],
    "tax_total": "number or null",
    "total": "grand total, number",
    "amount_paid": "number or null",
    "amount_due": "number or null",
    "payment_terms": "string or null",
    "notes": "string or null",
}

SYSTEM_PROMPT = f"""You are a meticulous accounts-payable assistant that extracts data from invoices and receipts.

Return ONLY one JSON object (no prose, no markdown) with exactly this structure:
{json.dumps(SCHEMA_SKELETON, indent=2)}

Rules:
- Copy values exactly as printed. Never invent, estimate or "fix" numbers to make totals add up.
- Use null for anything that is not present in the document.
- Numbers are plain JSON numbers with a dot as decimal separator and no currency symbols or thousands separators.
- Dates are ISO YYYY-MM-DD. If a numeric date is ambiguous (e.g. 03/04/2026), infer the order from the
  document's other dates, currency and locale.
- vendor = the party that issued the document and gets paid; buyer = the customer / bill-to party.
- One entry in line_items per billed line. amount is the line total as printed.
- discount_total is a positive number even if printed with a minus sign.
- taxes: one entry per tax rate / tax type printed (VAT, GST, sales tax ...). rate is a percentage (20 = 20%).
- currency: use the ISO 4217 code; map symbols only when unambiguous from context (€ = EUR, £ = GBP).
"""

_TEXT_INSTRUCTION = "Extract the invoice data from the document text below."
_IMAGE_INSTRUCTION = "Extract the invoice data from the attached document image(s). Pages are in order."


def build_messages(document: Document) -> list[Message]:
    """Build the initial chat messages for a text or image document."""
    system: Message = {"role": "system", "content": SYSTEM_PROMPT}
    if document.kind == "text":
        note = "\n(The text was truncated to fit the context window.)" if document.truncated else ""
        user: Message = {
            "role": "user",
            "content": f"{_TEXT_INSTRUCTION}{note}\n\n<document name={json.dumps(document.name)}>\n"
            f"{document.text}\n</document>",
        }
        return [system, user]

    parts: list[dict[str, Any]] = [{"type": "text", "text": _IMAGE_INSTRUCTION}]
    for image in document.images:
        encoded = base64.b64encode(image.data).decode("ascii")
        parts.append({"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"}})
    return [system, {"role": "user", "content": parts}]


def invalid_json_feedback(error: str) -> str:
    return (
        f"Your previous reply could not be parsed as JSON ({error}). "
        "Reply again with only the complete JSON object described in the instructions."
    )


def schema_feedback(errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "Your JSON does not match the required structure:\n"
        f"{listed}\n"
        "Re-read the document and reply with the complete, corrected JSON object only."
    )


def correction_feedback(issues: list[ValidationIssue]) -> str:
    listed = "\n".join(f"- [{i.code}] {i.message}" for i in issues)
    return (
        "The extracted numbers are not arithmetically consistent:\n"
        f"{listed}\n"
        "This usually means a value was misread (a digit, a decimal separator, a missed line item, or "
        "a field taken from the wrong row). Re-read the document carefully and reply with the complete, "
        "corrected JSON object only. If the document itself is inconsistent, keep the values exactly as "
        "printed — do not alter numbers just to make them add up."
    )
