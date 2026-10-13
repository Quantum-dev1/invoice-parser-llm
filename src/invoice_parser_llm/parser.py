"""The extraction pipeline: load → prompt → parse JSON → schema-validate → arithmetic-validate → self-correct."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .documents import Document, load_document
from .llm import LLMClient, LLMResponse, Message, OpenAICompatibleClient
from .prompts import build_messages, correction_feedback, invalid_json_feedback, schema_feedback
from .schema import Invoice
from .validation import ValidationIssue, validate_invoice

logger = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.75


class ParseResult(BaseModel):
    """Output of :meth:`InvoiceParser.parse`."""

    file: str | None = None
    invoice: Invoice | None = None
    confidence: float = Field(ge=0, le=1, description="Heuristic 0–1 score derived from validation results")
    needs_review: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    attempts: int = 0
    extraction: Literal["text", "vision"] = "text"
    model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of an LLM reply (tolerates code fences and leading/trailing prose)."""
    candidate = text.strip()
    fence = _FENCE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    try:
        obj = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg} at position {exc.pos}") from exc
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    # Tolerate {"invoice": {...}} wrappers.
    inner = obj.get("invoice")
    if len(obj) == 1 and isinstance(inner, dict):
        return inner
    return obj


def compute_confidence(issues: list[ValidationIssue], *, attempts: int, extraction: str, truncated: bool) -> float:
    """Heuristic confidence. Starts at 1.0 and subtracts for every signal of possible misreading:

    - 0.30 per arithmetic/consistency **error**
    - 0.05 per **warning** (missing fields, tax-rate rounding, …)
    - 0.05 per extra attempt the model needed
    - 0.05 when the document went through the vision path (scans/photos)
    - 0.10 when the text had to be truncated
    """
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    score = 1.0 - 0.30 * errors - 0.05 * warnings - 0.05 * max(attempts - 1, 0)
    if extraction == "vision":
        score -= 0.05
    if truncated:
        score -= 0.10
    return round(min(max(score, 0.0), 1.0), 2)


def _format_validation_error(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


class InvoiceParser:
    """Extract structured, validated invoice data with any OpenAI-compatible LLM.

    >>> parser = InvoiceParser()                      # configured from INVOICE_LLM_* env vars
    >>> result = parser.parse("invoice.pdf")
    >>> result.invoice.total, result.needs_review
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        vision_client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        s = self.settings
        if client is None:
            client = OpenAICompatibleClient(
                model=s.require_model(),
                base_url=s.base_url,
                api_key=s.api_key,
                timeout=s.timeout,
                max_retries=s.max_retries,
                json_mode=s.json_mode,
                temperature=s.temperature,
            )
        self.client = client
        if vision_client is None and s.vision_model and s.vision_model != getattr(client, "model", None):
            vision_client = OpenAICompatibleClient(
                model=s.vision_model,
                base_url=s.base_url,
                api_key=s.api_key,
                timeout=s.timeout,
                max_retries=s.max_retries,
                json_mode=s.json_mode,
                temperature=s.temperature,
            )
        self.vision_client = vision_client or client

    @classmethod
    def from_env(cls) -> InvoiceParser:
        return cls(settings=Settings.from_env())

    def load(self, source: str | Path | bytes, *, filename: str | None = None) -> Document:
        s = self.settings
        return load_document(
            source,
            filename=filename,
            max_pages=s.max_pages,
            min_text_chars=s.min_text_chars,
            image_dpi=s.image_dpi,
            max_text_chars=s.max_text_chars,
        )

    def parse(self, source: str | Path | bytes, *, filename: str | None = None) -> ParseResult:
        """Parse a file path or raw bytes (pass ``filename`` with bytes for nicer reporting)."""
        document = self.load(source, filename=filename)
        return self.parse_document(document)

    def parse_document(self, document: Document) -> ParseResult:
        extraction: Literal["text", "vision"] = "text" if document.kind == "text" else "vision"
        client = self.client if extraction == "text" else self.vision_client
        messages: list[Message] = build_messages(document)
        usage: dict[str, int] = {}

        best: tuple[tuple[int, int], Invoice, list[ValidationIssue]] | None = None
        last_problem = "no response"
        attempts = 0
        model_name: str | None = getattr(client, "model", None)

        for attempt in range(1, max(self.settings.max_attempts, 1) + 1):
            attempts = attempt
            response: LLMResponse = client.complete(messages)
            model_name = response.model or model_name
            for key, value in (response.usage or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value

            feedback: str
            try:
                data = extract_json_object(response.content)
            except ValueError as exc:
                last_problem = f"model reply was not valid JSON: {exc}"
                feedback = invalid_json_feedback(str(exc))
            else:
                try:
                    invoice = Invoice.model_validate(data)
                except ValidationError as exc:
                    errors = _format_validation_error(exc)
                    last_problem = "schema validation failed: " + "; ".join(errors[:5])
                    feedback = schema_feedback(errors)
                else:
                    issues = validate_invoice(invoice, self.settings.tolerance)
                    errors_found = [i for i in issues if i.severity == "error"]
                    rank = (len(errors_found), len(issues) - len(errors_found))
                    if best is None or rank <= best[0]:
                        best = (rank, invoice, issues)
                    if not errors_found:
                        break
                    last_problem = "arithmetic validation failed"
                    feedback = correction_feedback(errors_found)

            logger.info("attempt %d for %s needs correction: %s", attempt, document.name, last_problem)
            messages = [
                *messages,
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": feedback},
            ]

        if best is None:
            issue = ValidationIssue(code="extraction_failed", severity="error", message=last_problem)
            return ParseResult(
                file=document.name,
                invoice=None,
                confidence=0.0,
                needs_review=True,
                issues=[issue],
                attempts=attempts,
                extraction=extraction,
                model=model_name,
                warnings=document.notes,
                usage=usage,
            )

        _, invoice, issues = best
        confidence = compute_confidence(issues, attempts=attempts, extraction=extraction, truncated=document.truncated)
        return ParseResult(
            file=document.name,
            invoice=invoice,
            confidence=confidence,
            needs_review=any(i.severity == "error" for i in issues) or confidence < REVIEW_THRESHOLD,
            issues=issues,
            attempts=attempts,
            extraction=extraction,
            model=model_name,
            warnings=document.notes,
            usage=usage,
        )
