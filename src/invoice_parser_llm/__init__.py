"""invoice-parser-llm: invoices and receipts → validated JSON with any OpenAI-compatible LLM."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import ConfigError, Settings
from .documents import Document, UnsupportedDocumentError, load_document
from .llm import LLMClient, LLMError, LLMResponse, OpenAICompatibleClient
from .parser import InvoiceParser, ParseResult, compute_confidence, extract_json_object
from .schema import Invoice, LineItem, Party, TaxLine, invoice_json_schema, parse_decimal
from .validation import ValidationIssue, validate_invoice

__all__ = [
    "ConfigError",
    "Document",
    "Invoice",
    "InvoiceParser",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "LineItem",
    "OpenAICompatibleClient",
    "ParseResult",
    "Party",
    "Settings",
    "TaxLine",
    "UnsupportedDocumentError",
    "ValidationIssue",
    "__version__",
    "compute_confidence",
    "extract_json_object",
    "invoice_json_schema",
    "load_document",
    "parse_decimal",
    "validate_invoice",
]
