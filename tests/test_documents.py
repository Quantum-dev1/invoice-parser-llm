from __future__ import annotations

import io

import pytest
from PIL import Image

from invoice_parser_llm import UnsupportedDocumentError, load_document
from invoice_parser_llm.documents import sniff_format

from .conftest import EXAMPLES


def test_digital_pdf_text_extraction() -> None:
    doc = load_document(EXAMPLES / "initech-office-supplies.pdf")
    assert doc.kind == "text"
    assert doc.page_count == 1
    assert "2.757,35 EUR" in doc.text
    assert "Umbrella Robotics GmbH" in doc.text
    assert doc.text.startswith("--- page 1 ---")


def test_scanned_pdf_falls_back_to_images() -> None:
    doc = load_document(EXAMPLES / "initech-office-supplies-scanned.pdf", image_dpi=72)
    assert doc.kind == "image"
    assert len(doc.images) == 1
    assert doc.images[0].mime_type == "image/png"
    assert doc.notes


def test_min_text_chars_threshold_controls_fallback() -> None:
    # A digital PDF is treated as scanned if we demand more text per page than it has.
    doc = load_document(EXAMPLES / "acme-cloud-invoice.pdf", min_text_chars=100_000, image_dpi=50)
    assert doc.kind == "image"


def test_images_are_normalized_and_downscaled() -> None:
    big = Image.new("RGBA", (5000, 3000), (255, 255, 255, 255))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    doc = load_document(buf.getvalue(), filename="big.png")
    assert doc.kind == "image"
    out = Image.open(io.BytesIO(doc.images[0].data))
    assert max(out.size) == 2000
    assert out.mode == "RGB"


def test_photo_jpeg() -> None:
    doc = load_document(EXAMPLES / "initech-office-supplies-photo.jpg")
    assert doc.kind == "image"
    assert doc.source_format == "jpeg"


def test_text_files_pass_through_and_truncate() -> None:
    doc = load_document(b"INVOICE 1\nTotal 10.00", filename="inv.txt")
    assert doc.kind == "text" and "Total 10.00" in doc.text
    long = load_document(b"x" * 100, filename="inv.txt", max_text_chars=10)
    assert long.truncated and len(long.text) == 10


def test_page_limit_note(tmp_path: object) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    for i in range(3):
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, f"Invoice page {i + 1} with enough text to count as a digital document page.")
    doc = load_document(bytes(pdf.output()), filename="multi.pdf", max_pages=2)
    assert doc.page_count == 3
    assert "--- page 2 ---" in doc.text and "--- page 3 ---" not in doc.text
    assert any("first 2 of 3" in n for n in doc.notes)


@pytest.mark.parametrize(
    ("data", "fmt"),
    [
        (b"%PDF-1.7", "pdf"),
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xff\xd8\xff\xe0", "jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBP", "webp"),
    ],
)
def test_sniff_format(data: bytes, fmt: str) -> None:
    assert sniff_format(data, "x") == fmt


def test_unsupported_files() -> None:
    with pytest.raises(UnsupportedDocumentError):
        load_document(b"PK\x03\x04\x00\x00\xff\xfe binary zip", filename="invoice.docx")
    with pytest.raises(UnsupportedDocumentError):
        load_document(b"%PDF-1.4 garbage that is not a pdf", filename="broken.pdf")
