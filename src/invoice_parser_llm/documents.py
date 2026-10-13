"""Load PDFs and images into either extracted text or page images for a vision model.

Strategy:

1. **Digital PDFs** → text via ``pdfplumber`` (layout-aware), falling back to ``pypdf`` if pdfplumber fails.
2. **Scanned PDFs** (too little text per page) → pages rendered to PNG and sent to a vision model.
3. **Images** (PNG, JPEG, WebP, TIFF, GIF, BMP) → normalized to PNG/JPEG and sent to a vision model.
4. **Plain text** files are passed through.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp", ".txt")
_MAX_IMAGE_SIDE = 2000


class UnsupportedDocumentError(ValueError):
    """The input is not a PDF, image or text file, or is unreadable."""


@dataclass
class PageImage:
    data: bytes
    mime_type: str


@dataclass
class Document:
    """A loaded document, ready to be sent to the LLM."""

    name: str
    kind: Literal["text", "image"]
    text: str = ""
    images: list[PageImage] = field(default_factory=list)
    page_count: int = 1
    source_format: str = "pdf"
    truncated: bool = False
    notes: list[str] = field(default_factory=list)


def sniff_format(data: bytes, filename: str | None = None) -> str:
    """Detect the file format from magic bytes (falling back to the extension)."""
    head = data[:16]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:2] == b"BM":
        return "bmp"
    ext = Path(filename).suffix.lower() if filename else ""
    if ext == ".txt":
        return "text"
    try:
        data[:4096].decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        if ext in ("", ".txt", ".text"):
            return "text"
    raise UnsupportedDocumentError(f"Unsupported or unrecognized file type: {filename or '<bytes>'}")


def _image_to_payload(image: Image.Image) -> PageImage:
    """Downscale huge images and encode as PNG (or JPEG for photos) to keep requests small."""
    img = image
    if max(img.size) > _MAX_IMAGE_SIDE:
        img = img.copy()
        img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    # PNG keeps text crisp; switch to JPEG when the PNG would be very large (photos of receipts).
    img.save(buf, format="PNG", optimize=True)
    if buf.tell() > 4 * 1024 * 1024:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return PageImage(buf.getvalue(), "image/jpeg")
    return PageImage(buf.getvalue(), "image/png")


def _pdf_text_pdfplumber(data: bytes, max_pages: int) -> tuple[list[str], int]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = [(page.extract_text() or "") for page in pdf.pages[:max_pages]]
        return pages, len(pdf.pages)


def _pdf_text_pypdf(data: bytes, max_pages: int) -> tuple[list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "") for page in reader.pages[:max_pages]]
    return pages, len(reader.pages)


def _render_pdf_pages(data: bytes, max_pages: int, dpi: int) -> list[PageImage]:
    import pdfplumber

    images: list[PageImage] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:max_pages]:
            rendered = page.to_image(resolution=dpi).original
            images.append(_image_to_payload(rendered))
    return images


def load_document(
    source: str | Path | bytes,
    *,
    filename: str | None = None,
    max_pages: int = 5,
    min_text_chars: int = 40,
    image_dpi: int = 150,
    max_text_chars: int = 60_000,
) -> Document:
    """Load a PDF / image / text file from a path or raw bytes.

    ``min_text_chars`` is the minimum average number of non-whitespace characters per page for a PDF
    to be treated as digital; below that it is considered scanned and rendered to images.
    """
    if isinstance(source, bytes):
        data = source
        name = filename or "document"
    else:
        path = Path(source)
        data = path.read_bytes()
        name = filename or path.name

    fmt = sniff_format(data, name)

    if fmt == "text":
        text = data.decode("utf-8", errors="replace")
        truncated = len(text) > max_text_chars
        return Document(name=name, kind="text", text=text[:max_text_chars], source_format="text", truncated=truncated)

    if fmt != "pdf":
        try:
            with Image.open(io.BytesIO(data)) as img:
                frames = getattr(img, "n_frames", 1)
                images = []
                for i in range(min(frames, max_pages)):
                    img.seek(i)
                    images.append(_image_to_payload(img.copy()))
        except Exception as exc:
            raise UnsupportedDocumentError(f"Could not read image {name}: {exc}") from exc
        return Document(name=name, kind="image", images=images, page_count=len(images), source_format=fmt)

    notes: list[str] = []
    try:
        pages, page_count = _pdf_text_pdfplumber(data, max_pages)
    except Exception as exc:
        logger.warning("pdfplumber failed on %s (%s); falling back to pypdf", name, exc)
        notes.append("pdfplumber failed; text extracted with pypdf")
        try:
            pages, page_count = _pdf_text_pypdf(data, max_pages)
        except Exception as exc2:
            raise UnsupportedDocumentError(f"Could not read PDF {name}: {exc2}") from exc2

    if page_count > max_pages:
        notes.append(f"only the first {max_pages} of {page_count} pages were processed")

    meaningful = sum(len("".join(p.split())) for p in pages)
    if pages and meaningful / len(pages) >= min_text_chars:
        text = "\n\n".join(f"--- page {i + 1} ---\n{p}" for i, p in enumerate(pages))
        truncated = len(text) > max_text_chars
        return Document(
            name=name,
            kind="text",
            text=text[:max_text_chars],
            page_count=page_count,
            source_format="pdf",
            truncated=truncated,
            notes=notes,
        )

    notes.append("PDF has little or no text layer; pages rendered to images for a vision model")
    try:
        images = _render_pdf_pages(data, max_pages, image_dpi)
    except Exception as exc:
        raise UnsupportedDocumentError(f"Could not render scanned PDF {name}: {exc}") from exc
    return Document(name=name, kind="image", images=images, page_count=page_count, source_format="pdf", notes=notes)
