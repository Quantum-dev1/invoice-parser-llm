"""FastAPI service. Run with ``invoice-parser serve`` or ``uvicorn invoice_parser_llm.api:app``.

Environment:
    INVOICE_LLM_*            LLM configuration (see config.py / README)
    INVOICE_API_TOKEN        optional; when set, requests need ``Authorization: Bearer <token>``
    INVOICE_MAX_UPLOAD_MB    maximum upload size in MB (default 15)
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from . import __version__
from .config import ConfigError, Settings
from .documents import UnsupportedDocumentError
from .llm import LLMError
from .parser import InvoiceParser, ParseResult
from .schema import Invoice, invoice_json_schema
from .validation import ValidationIssue, validate_invoice


@lru_cache(maxsize=1)
def get_parser() -> InvoiceParser:
    """Build the parser once from the environment (overridable in tests via dependency_overrides)."""
    return InvoiceParser(settings=Settings.from_env())


def _max_upload_bytes() -> int:
    return int(float(os.environ.get("INVOICE_MAX_UPLOAD_MB", "15")) * 1024 * 1024)


def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.environ.get("INVOICE_API_TOKEN")
    if not expected:
        return
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token")


def resolve_parser() -> InvoiceParser:
    try:
        return get_parser()
    except ConfigError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def create_app() -> FastAPI:
    api = FastAPI(
        title="invoice-parser-llm",
        version=__version__,
        description="Turn invoice and receipt PDFs/images into validated JSON with any OpenAI-compatible LLM.",
    )

    @api.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, Any]:
        settings = Settings.from_env()
        return {"status": "ok", "version": __version__, "model_configured": bool(settings.model)}

    @api.get("/v1/schema", tags=["meta"])
    def schema() -> dict[str, Any]:
        return invoice_json_schema()

    @api.post("/v1/parse", response_model=ParseResult, tags=["parse"], dependencies=[Depends(require_token)])
    async def parse(
        file: Annotated[UploadFile, File(description="PDF or image (PNG, JPEG, WebP, TIFF)")],
        parser: Annotated[InvoiceParser, Depends(resolve_parser)],
    ) -> ParseResult:
        limit = _max_upload_bytes()
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"file larger than {limit} bytes")
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
        try:
            return await run_in_threadpool(parser.parse, data, filename=file.filename or "upload")
        except UnsupportedDocumentError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    @api.post(
        "/v1/validate",
        response_model=list[ValidationIssue],
        tags=["parse"],
        dependencies=[Depends(require_token)],
    )
    def validate(invoice: Invoice, tolerance: float = 0.01) -> list[ValidationIssue]:
        """Run only the arithmetic checks on an invoice you already have (no LLM call)."""
        return validate_invoice(invoice, str(tolerance))

    return api


app = create_app()
