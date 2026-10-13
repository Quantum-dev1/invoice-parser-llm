# Changelog

## 0.1.0 — unreleased

Initial release.

- Extraction pipeline: PDF text (pdfplumber, pypdf fallback), scanned-PDF and image → vision model.
- Pydantic v2 invoice schema with lenient money/date/currency parsing.
- Deterministic arithmetic validation and a self-correction loop with best-attempt selection.
- Confidence score and `needs_review` flag.
- OpenAI-compatible HTTP client with retries and JSON-mode fallback.
- CLI: `parse`, `batch` (JSONL + CSVs), `eval`, `validate`, `schema`, `serve`.
- FastAPI service and Docker image.
- Synthetic sample invoices with ground truth.
