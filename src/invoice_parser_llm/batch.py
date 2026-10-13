"""Batch processing: a folder of documents → JSONL / CSV."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from .documents import SUPPORTED_EXTENSIONS
from .parser import InvoiceParser, ParseResult

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    path: Path
    result: ParseResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None and self.result.invoice is not None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"file": str(self.path)}
        if self.result is not None:
            record.update(self.result.model_dump(mode="json", exclude={"file"}))
        if self.error:
            record["error"] = self.error
        return record


def find_documents(folder: str | Path, *, recursive: bool = False) -> list[Path]:
    """All supported files in ``folder`` (sorted for stable output)."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def parse_many(
    parser: InvoiceParser,
    paths: Iterable[Path],
    *,
    workers: int = 4,
    on_done: Callable[[BatchItem], None] | None = None,
) -> Iterator[BatchItem]:
    """Parse files concurrently, yielding results as they complete. One failure never stops the batch."""

    def run(path: Path) -> BatchItem:
        try:
            return BatchItem(path=path, result=parser.parse(path))
        except Exception as exc:
            logger.exception("failed to parse %s", path)
            return BatchItem(path=path, error=f"{type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = [pool.submit(run, p) for p in paths]
        for future in as_completed(futures):
            item = future.result()
            if on_done:
                on_done(item)
            yield item


def write_jsonl(items: Iterable[BatchItem], fp: IO[str]) -> int:
    count = 0
    for item in items:
        fp.write(json.dumps(item.to_record(), ensure_ascii=False) + "\n")
        count += 1
    return count


SUMMARY_COLUMNS = [
    "file",
    "status",
    "needs_review",
    "confidence",
    "document_type",
    "invoice_number",
    "issue_date",
    "due_date",
    "currency",
    "vendor_name",
    "vendor_tax_id",
    "buyer_name",
    "subtotal",
    "discount_total",
    "shipping_total",
    "tax_total",
    "total",
    "amount_due",
    "line_items",
    "issues",
    "error",
]

ITEM_COLUMNS = [
    "file",
    "invoice_number",
    "vendor_name",
    "currency",
    "line",
    "description",
    "sku",
    "quantity",
    "unit",
    "unit_price",
    "tax_rate",
    "amount",
]


def summary_row(item: BatchItem) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(SUMMARY_COLUMNS, "")
    row["file"] = str(item.path)
    row["error"] = item.error or ""
    result = item.result
    if result is None:
        row["status"] = "error"
        return row
    row["status"] = "ok" if result.invoice else "failed"
    row["needs_review"] = result.needs_review
    row["confidence"] = result.confidence
    row["issues"] = ";".join(i.code for i in result.issues)
    inv = result.invoice
    if inv is None:
        return row
    data = inv.model_dump(mode="json")
    for key in (
        "document_type",
        "invoice_number",
        "issue_date",
        "due_date",
        "currency",
        "subtotal",
        "discount_total",
        "shipping_total",
        "tax_total",
        "total",
        "amount_due",
    ):
        row[key] = "" if data.get(key) is None else data[key]
    if inv.tax_total is None and inv.taxes:
        row["tax_total"] = sum(float(t.amount) for t in inv.taxes)
    row["vendor_name"] = inv.vendor.name or ""
    row["vendor_tax_id"] = inv.vendor.tax_id or ""
    row["buyer_name"] = (inv.buyer.name if inv.buyer else None) or ""
    row["line_items"] = len(inv.line_items)
    return row


def write_summary_csv(items: Iterable[BatchItem], fp: IO[str]) -> int:
    writer = csv.DictWriter(fp, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    count = 0
    for item in items:
        writer.writerow(summary_row(item))
        count += 1
    return count


def write_items_csv(items: Iterable[BatchItem], fp: IO[str]) -> int:
    """One row per line item — convenient for spreadsheets and accounting imports."""
    writer = csv.DictWriter(fp, fieldnames=ITEM_COLUMNS)
    writer.writeheader()
    count = 0
    for item in items:
        inv = item.result.invoice if item.result else None
        if inv is None:
            continue
        for n, line in enumerate(inv.line_items, start=1):
            d = line.model_dump(mode="json")
            writer.writerow(
                {
                    "file": str(item.path),
                    "invoice_number": inv.invoice_number or "",
                    "vendor_name": inv.vendor.name or "",
                    "currency": inv.currency or "",
                    "line": n,
                    **{k: ("" if d.get(k) is None else d[k]) for k in ITEM_COLUMNS[5:]},
                }
            )
            count += 1
    return count
