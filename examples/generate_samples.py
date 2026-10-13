"""Generate synthetic sample invoices for tests and demos.

All companies, people, addresses and tax IDs are fictional. Every document carries a
"SAMPLE DOCUMENT" footer. Ground-truth JSON for each document is written to examples/expected/.

    pip install -e ".[dev]"        # needs fpdf2
    python examples/generate_samples.py

Outputs (examples/invoices/):
    acme-cloud-invoice.pdf          digital PDF, GBP, 20% VAT, fractional hours
    northwind-receipt.pdf           narrow till receipt, USD, sales tax, paid by card
    initech-office-supplies.pdf     EUR, European number format, discount + shipping + two VAT rates
    initech-office-supplies-photo.jpg       "photographed" version (vision path)
    initech-office-supplies-scanned.pdf     image-only PDF (scanned-PDF → vision fallback)
"""

from __future__ import annotations

import io
import json
import random
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pdfplumber
from fpdf import FPDF
from PIL import Image, ImageFilter

from invoice_parser_llm import Invoice, validate_invoice

HERE = Path(__file__).resolve().parent
OUT = HERE / "invoices"
EXPECTED = HERE / "expected"
FIXED_DATE = datetime(2026, 10, 1, tzinfo=UTC)
FOOTER = "SAMPLE DOCUMENT - fictional companies, generated for testing invoice-parser-llm."

D = Decimal


def money_en(value: Decimal, symbol: str = "") -> str:
    return f"{symbol}{value:,.2f}"


def money_eu(value: Decimal) -> str:
    s = f"{value:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") + " EUR"


def new_pdf(width: float = 210, height: float = 297) -> FPDF:
    pdf = FPDF(unit="mm", format=(width, height))
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_producer("invoice-parser-llm sample generator")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    return pdf


def table(pdf: FPDF, headers: list[str], rows: list[list[str]], widths: list[float], aligns: list[str]) -> None:
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(235, 239, 245)
    for h, w, a in zip(headers, widths, aligns, strict=True):
        pdf.cell(w, 8, h, border=0, align=a, fill=True)
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9)
    for row in rows:
        for v, w, a in zip(row, widths, aligns, strict=True):
            pdf.cell(w, 7, v, border="B", align=a)
        pdf.ln(7)


def totals(pdf: FPDF, lines: list[tuple[str, str]], x: float = 120, bold_last: bool = True) -> None:
    pdf.ln(4)
    for i, (label, value) in enumerate(lines):
        last = bold_last and i == len(lines) - 1
        pdf.set_font("Helvetica", "B" if last else "", 10 if last else 9)
        pdf.set_x(x)
        pdf.cell(45, 7, label)
        pdf.cell(25, 7, value, align="R")
        pdf.ln(7)


def footer(pdf: FPDF, y: float) -> None:
    pdf.set_y(y)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 5, FOOTER, align="C")
    pdf.set_text_color(0, 0, 0)


# ---------------------------------------------------------------------------------------------------------
# 1. Acme Cloud Services — GBP, 20% VAT
# ---------------------------------------------------------------------------------------------------------
def acme() -> tuple[bytes, dict[str, Any]]:
    items = [
        ("Cloud hosting - Pro plan (September)", D("3"), "seat", D("249.00")),
        ("Managed PostgreSQL cluster", D("1"), "month", D("180.00")),
        ("Engineering support", D("6.5"), "h", D("95.00")),
    ]
    lines = [(d, q, u, p, (q * p).quantize(D("0.01"))) for d, q, u, p in items]
    subtotal = sum((amt for *_, amt in lines), D(0))
    vat = (subtotal * D("0.20")).quantize(D("0.01"))
    total = subtotal + vat

    pdf = new_pdf()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 12, "Acme Cloud Services Ltd.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(12)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        90,
        5,
        "42 Example Lane\nSpringfield SP1 2AB\nUnited Kingdom\nVAT No. GB000000042\nbilling@acme-cloud.example",
    )
    pdf.set_xy(130, 12)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "INVOICE", align="R")
    pdf.set_xy(130, 24)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        70,
        5,
        "Invoice no.: INV-2026-0042\nInvoice date: 30 Sep 2026\nDue date: 30 Oct 2026\nTerms: Net 30",
        align="R",
    )
    pdf.set_xy(10, 62)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "Bill to")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(90, 5, "Globex Analytics plc\n7 Sample Street\nRiverton RV3 9XY\nUnited Kingdom")
    pdf.ln(6)
    table(
        pdf,
        ["Description", "Qty", "Unit", "Unit price", "Amount"],
        [[d, f"{q.normalize()}", u, money_en(p, "£"), money_en(a, "£")] for d, q, u, p, a in lines],
        [90, 18, 18, 32, 32],
        ["L", "R", "C", "R", "R"],
    )
    totals(
        pdf,
        [
            ("Subtotal", money_en(subtotal, "£")),
            ("VAT 20%", money_en(vat, "£")),
            ("Total due (GBP)", money_en(total, "£")),
        ],
    )
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(
        0,
        4.5,
        "Please pay by bank transfer within 30 days quoting the invoice number.\nThank you for your business!",
    )
    footer(pdf, 285)

    expected = {
        "document_type": "invoice",
        "invoice_number": "INV-2026-0042",
        "issue_date": "2026-09-30",
        "due_date": "2026-10-30",
        "currency": "GBP",
        "vendor": {
            "name": "Acme Cloud Services Ltd.",
            "address": "42 Example Lane, Springfield SP1 2AB, United Kingdom",
            "tax_id": "GB000000042",
            "email": "billing@acme-cloud.example",
        },
        "buyer": {
            "name": "Globex Analytics plc",
            "address": "7 Sample Street, Riverton RV3 9XY, United Kingdom",
        },
        "line_items": [
            {"description": d, "quantity": str(q), "unit": u, "unit_price": str(p), "amount": str(a)}
            for d, q, u, p, a in lines
        ],
        "subtotal": str(subtotal),
        "taxes": [{"name": "VAT", "rate": "20", "base": str(subtotal), "amount": str(vat)}],
        "tax_total": str(vat),
        "total": str(total),
        "amount_due": str(total),
        "payment_terms": "Net 30",
    }
    return bytes(pdf.output()), expected


# ---------------------------------------------------------------------------------------------------------
# 2. Northwind Coffee Roasters — USD till receipt
# ---------------------------------------------------------------------------------------------------------
def northwind() -> tuple[bytes, dict[str, Any]]:
    items = [
        ("Cappuccino", D("2"), D("4.75")),
        ("Blueberry muffin", D("1"), D("3.95")),
        ("Single-origin beans 340g", D("1"), D("18.00")),
    ]
    lines = [(d, q, p, q * p) for d, q, p in items]
    subtotal = sum((a for *_, a in lines), D(0))
    tax = (subtotal * D("0.0825")).quantize(D("0.01"))
    total = subtotal + tax

    pdf = new_pdf(80, 170)
    pdf.set_margins(6, 6, 6)
    pdf.set_y(8)
    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 6, "NORTHWIND COFFEE ROASTERS", align="C")
    pdf.ln(6)
    pdf.set_font("Courier", "", 8)
    for line in ("1200 Harbor Blvd, Bayview", "Tel (555) 010-2030", "Tax ID 00-0000000"):
        pdf.cell(0, 4, line, align="C")
        pdf.ln(4)
    pdf.ln(2)
    pdf.cell(0, 4, "Receipt #R-58213")
    pdf.ln(4)
    pdf.cell(0, 4, "Oct 2, 2026  08:14   Register 2")
    pdf.ln(5)
    pdf.cell(0, 3, "-" * 40)
    pdf.ln(4)
    for d, q, p, a in lines:
        pdf.cell(0, 4, d)
        pdf.ln(4)
        pdf.cell(40, 4, f"  {q} x {p:.2f}")
        pdf.cell(0, 4, f"{a:.2f}", align="R")
        pdf.ln(4)
    pdf.cell(0, 3, "-" * 40)
    pdf.ln(4)
    for label, value in (
        ("SUBTOTAL", subtotal),
        ("SALES TAX 8.25%", tax),
        ("TOTAL USD", total),
        ("VISA ****4242", total),
        ("CHANGE DUE", D("0.00")),
    ):
        pdf.set_font("Courier", "B" if label == "TOTAL USD" else "", 8)
        pdf.cell(40, 4.5, label)
        pdf.cell(0, 4.5, f"{value:.2f}", align="R")
        pdf.ln(4.5)
    pdf.ln(3)
    pdf.set_font("Courier", "", 8)
    pdf.cell(0, 4, "Thank you - see you tomorrow!", align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 5)
    pdf.multi_cell(0, 3, FOOTER, align="C")

    expected = {
        "document_type": "receipt",
        "invoice_number": "R-58213",
        "issue_date": "2026-10-02",
        "currency": "USD",
        "vendor": {
            "name": "Northwind Coffee Roasters",
            "address": "1200 Harbor Blvd, Bayview",
            "tax_id": "00-0000000",
            "phone": "(555) 010-2030",
        },
        "buyer": None,
        "line_items": [
            {"description": d, "quantity": str(q), "unit_price": str(p), "amount": str(a)} for d, q, p, a in lines
        ],
        "subtotal": str(subtotal),
        "taxes": [{"name": "Sales tax", "rate": "8.25", "base": str(subtotal), "amount": str(tax)}],
        "tax_total": str(tax),
        "total": str(total),
        "amount_paid": str(total),
        "amount_due": "0.00",
    }
    return bytes(pdf.output()), expected


# ---------------------------------------------------------------------------------------------------------
# 3. Initech Office Supplies — EUR, European formatting, discount, shipping, two VAT rates
# ---------------------------------------------------------------------------------------------------------
def initech() -> tuple[bytes, dict[str, Any]]:
    items = [
        ("Ergonomic office chair", "CH-200", D("4"), D("289.00"), D("21")),
        ("Standing desk frame", "DK-415", D("2"), D("415.00"), D("21")),
        ("Printer paper A4, box of 5 reams", "PP-005", D("10"), D("24.90"), D("21")),
        ("Office coffee beans, 6 kg", "CF-006", D("1"), D("120.00"), D("9")),
    ]
    lines = [(d, sku, q, p, r, q * p) for d, sku, q, p, r in items]
    subtotal = sum((a for *_, a in lines), D(0))
    discount = ((lines[0][5] + lines[1][5]) * D("0.05")).quantize(D("0.01"))  # 5% on furniture
    shipping = D("35.00")
    base21 = sum((a for *_, r, a in lines if r == 21), D(0)) - discount + shipping
    base9 = sum((a for *_, r, a in lines if r == 9), D(0))
    vat21 = (base21 * D("0.21")).quantize(D("0.01"))
    vat9 = (base9 * D("0.09")).quantize(D("0.01"))
    total = subtotal - discount + shipping + vat21 + vat9

    pdf = new_pdf()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "INITECH OFFICE SUPPLIES B.V.")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "Sample Canal 10 | 1011 AB Lakeside | VAT ID NL000000000B01 | orders@initech-supplies.example")
    pdf.ln(14)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(95, 5, "Customer")
    pdf.cell(0, 5, "Invoice")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 9)
    y = pdf.get_y()
    pdf.multi_cell(
        95,
        5,
        "Umbrella Robotics GmbH\nAttn. Purchasing\nExample Allee 5\n10115 Northgate\nVAT ID DE000000000",
    )
    pdf.set_xy(105, y)
    pdf.multi_cell(
        95,
        5,
        "Number: 2026-1187\nDate: 05.10.2026\nPayable by: 19.10.2026 (14 days net)\nCustomer no.: C-0815",
    )
    pdf.ln(8)
    table(
        pdf,
        ["Item", "Description", "Qty", "Unit price", "VAT", "Amount"],
        [[sku, d, str(q), money_eu(p), f"{r}%", money_eu(a)] for d, sku, q, p, r, a in lines],
        [20, 70, 14, 30, 16, 40],
        ["L", "L", "R", "R", "R", "R"],
    )
    totals(
        pdf,
        [
            ("Net total", money_eu(subtotal)),
            ("Discount 5% furniture", "-" + money_eu(discount)),
            ("Shipping", money_eu(shipping)),
            (f"VAT 21% on {money_eu(base21)}", money_eu(vat21)),
            (f"VAT 9% on {money_eu(base9)}", money_eu(vat9)),
            ("Invoice total", money_eu(total)),
        ],
        x=95,
    )
    pdf.ln(6)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(0, 4.5, "Bank: Example Bank N.V. | IBAN XX00 0000 0000 0000 00 | Reference: 2026-1187")
    footer(pdf, 285)

    expected = {
        "document_type": "invoice",
        "invoice_number": "2026-1187",
        "issue_date": "2026-10-05",
        "due_date": "2026-10-19",
        "currency": "EUR",
        "vendor": {
            "name": "Initech Office Supplies B.V.",
            "address": "Sample Canal 10, 1011 AB Lakeside",
            "tax_id": "NL000000000B01",
            "email": "orders@initech-supplies.example",
        },
        "buyer": {
            "name": "Umbrella Robotics GmbH",
            "address": "Example Allee 5, 10115 Northgate",
            "tax_id": "DE000000000",
        },
        "line_items": [
            {
                "description": d,
                "sku": sku,
                "quantity": str(q),
                "unit_price": str(p),
                "tax_rate": str(r),
                "amount": str(a),
            }
            for d, sku, q, p, r, a in lines
        ],
        "subtotal": str(subtotal),
        "discount_total": str(discount),
        "shipping_total": str(shipping),
        "taxes": [
            {"name": "VAT", "rate": "21", "base": str(base21), "amount": str(vat21)},
            {"name": "VAT", "rate": "9", "base": str(base9), "amount": str(vat9)},
        ],
        "tax_total": str(vat21 + vat9),
        "total": str(total),
        "amount_due": str(total),
        "payment_terms": "14 days net",
    }
    return bytes(pdf.output()), expected


def scan(pdf_bytes: bytes) -> tuple[bytes, bytes]:
    """Fake a phone photo / scan: render, tilt slightly, add grain. Returns (jpeg, image-only pdf)."""
    rng = random.Random(42)
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as doc:
        page = doc.pages[0].to_image(resolution=130).original.convert("L")
    tilted = page.rotate(0.8, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    noise = Image.effect_noise(tilted.size, 18)
    noisy = Image.blend(tilted, noise, 0.06).filter(ImageFilter.GaussianBlur(0.4))
    tinted = Image.merge("RGB", (noisy, noisy, noisy.point(lambda v: int(v * 0.96))))
    # Crop a few random pixels off the edges like an imperfect scan.
    w, h = tinted.size
    tinted = tinted.crop((rng.randint(0, 8), rng.randint(0, 8), w - rng.randint(0, 8), h - rng.randint(0, 8)))

    jpeg = io.BytesIO()
    tinted.save(jpeg, format="JPEG", quality=72, optimize=True)
    pdf = new_pdf()
    pdf.image(io.BytesIO(jpeg.getvalue()), x=0, y=0, w=210, h=297)
    return jpeg.getvalue(), bytes(pdf.output())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EXPECTED.mkdir(parents=True, exist_ok=True)
    for name, build in (
        ("acme-cloud-invoice", acme),
        ("northwind-receipt", northwind),
        ("initech-office-supplies", initech),
    ):
        pdf_bytes, expected = build()
        invoice = Invoice.model_validate(expected)
        issues = validate_invoice(invoice)
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"{name}: ground truth is inconsistent: {errors}"
        (OUT / f"{name}.pdf").write_bytes(pdf_bytes)
        (EXPECTED / f"{name}.json").write_text(
            json.dumps(invoice.model_dump(mode="json", exclude_none=True), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote invoices/{name}.pdf and expected/{name}.json")
        if name == "initech-office-supplies":
            photo, scanned_pdf = scan(pdf_bytes)
            (OUT / f"{name}-photo.jpg").write_bytes(photo)
            (OUT / f"{name}-scanned.pdf").write_bytes(scanned_pdf)
            print(f"wrote invoices/{name}-photo.jpg and invoices/{name}-scanned.pdf")


if __name__ == "__main__":
    main()
