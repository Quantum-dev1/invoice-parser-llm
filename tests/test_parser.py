from __future__ import annotations

import copy
from decimal import Decimal
from typing import Any

import pytest

from invoice_parser_llm import LLMError, Settings, compute_confidence, extract_json_object
from invoice_parser_llm.validation import ValidationIssue

from .conftest import EXAMPLES, FakeLLM, make_parser


def test_extract_json_object_tolerates_fences_and_prose() -> None:
    assert extract_json_object('Sure! ```json\n{"total": 1}\n``` hope that helps') == {"total": 1}
    assert extract_json_object('Here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert extract_json_object('{"invoice": {"total": 3}}') == {"total": 3}
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object("I could not read the document.")
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_json_object('{"total": 1,}')


def test_happy_path_text_pdf(acme_expected: dict[str, Any], settings: Settings) -> None:
    fake = FakeLLM(acme_expected)
    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")

    assert result.invoice is not None
    assert result.invoice.total == Decimal("1853.40")
    assert result.issues == []
    assert result.confidence == 1.0
    assert result.needs_review is False
    assert result.attempts == 1
    assert result.extraction == "text"
    assert result.model == "fake-model"
    assert result.usage == {"prompt_tokens": 100, "completion_tokens": 50}

    # The prompt contains the extracted PDF text, not an image.
    system, user = fake.calls[0]
    assert system["role"] == "system" and "Never invent" in system["content"]
    assert "INV-2026-0042" in user["content"] and "Globex Analytics plc" in user["content"]


def test_self_correction_fixes_arithmetic(acme_expected: dict[str, Any], settings: Settings) -> None:
    misread = copy.deepcopy(acme_expected)
    misread["line_items"][2]["amount"] = 671.50
    misread["subtotal"] = 1598.50
    fake = FakeLLM(misread, acme_expected)

    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")

    assert result.attempts == 2
    assert result.issues == []
    assert result.needs_review is False
    assert result.confidence == 0.95  # one extra attempt
    # The correction turn tells the model exactly what did not add up.
    feedback = fake.calls[1][-1]["content"]
    assert "line_amount_mismatch" in feedback and "617.50" in feedback
    assert "do not alter numbers just to make them add up" in feedback
    assert fake.calls[1][-2]["role"] == "assistant"


def test_persistent_inconsistency_is_flagged_not_hidden(acme_expected: dict[str, Any], settings: Settings) -> None:
    wrong = copy.deepcopy(acme_expected)
    wrong["total"] = 1835.40
    wrong["amount_due"] = 1835.40
    fake = FakeLLM(wrong, wrong, wrong)

    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")

    assert result.attempts == 3
    assert result.invoice is not None
    assert [i.code for i in result.issues] == ["total_mismatch"]
    assert result.needs_review is True
    assert result.confidence == pytest.approx(0.6)


def test_best_attempt_wins(acme_expected: dict[str, Any], settings: Settings) -> None:
    one_error = copy.deepcopy(acme_expected)
    one_error["total"] = 1.0
    one_error["amount_due"] = None
    two_errors = copy.deepcopy(one_error)
    two_errors["subtotal"] = 2.0
    fake = FakeLLM(one_error, two_errors, two_errors)
    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")
    assert result.invoice is not None
    assert result.invoice.subtotal == Decimal("1544.50")  # from the first (one-error) attempt


def test_recovers_from_invalid_json_and_schema_errors(acme_expected: dict[str, Any], settings: Settings) -> None:
    fake = FakeLLM("Sorry, here it is: {not json", {"issue_date": "03/04/2026", "total": 1}, acme_expected)
    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")
    assert result.attempts == 3
    assert result.issues == []
    assert "could not be parsed as JSON" in fake.calls[1][-1]["content"]
    assert "issue_date" in fake.calls[2][-1]["content"]


def test_gives_up_cleanly(settings: Settings) -> None:
    fake = FakeLLM("nope", "still nope", "no")
    result = make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")
    assert result.invoice is None
    assert result.needs_review is True
    assert result.confidence == 0.0
    assert result.issues[0].code == "extraction_failed"


def test_llm_errors_propagate(settings: Settings) -> None:
    fake = FakeLLM(LLMError("HTTP 401"))
    with pytest.raises(LLMError):
        make_parser(fake, settings).parse(EXAMPLES / "acme-cloud-invoice.pdf")


@pytest.mark.parametrize("filename", ["initech-office-supplies-photo.jpg", "initech-office-supplies-scanned.pdf"])
def test_images_and_scanned_pdfs_use_vision(
    filename: str, initech_expected: dict[str, Any], settings: Settings
) -> None:
    text_llm = FakeLLM(model="text-model")
    vision_llm = FakeLLM(initech_expected, model="vision-model")
    from invoice_parser_llm import InvoiceParser

    parser = InvoiceParser(client=text_llm, vision_client=vision_llm, settings=settings)
    result = parser.parse(EXAMPLES / filename)

    assert text_llm.calls == []
    assert result.extraction == "vision"
    assert result.model == "vision-model"
    assert result.confidence == 0.95
    content = vision_llm.calls[0][1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/")
    if filename.endswith(".pdf"):
        assert any("rendered to images" in w for w in result.warnings)


def test_parse_bytes_with_filename(northwind_expected: dict[str, Any], settings: Settings) -> None:
    fake = FakeLLM(northwind_expected)
    data = (EXAMPLES / "northwind-receipt.pdf").read_bytes()
    result = make_parser(fake, settings).parse(data, filename="upload.pdf")
    assert result.file == "upload.pdf"
    assert result.invoice is not None and result.invoice.document_type == "receipt"


def test_confidence_heuristic() -> None:
    err = ValidationIssue(code="x", severity="error", message="")
    warn = ValidationIssue(code="y", severity="warning", message="")
    assert compute_confidence([], attempts=1, extraction="text", truncated=False) == 1.0
    assert compute_confidence([warn], attempts=1, extraction="text", truncated=False) == 0.95
    assert compute_confidence([err, warn], attempts=3, extraction="vision", truncated=True) == 0.4
    assert compute_confidence([err] * 5, attempts=1, extraction="text", truncated=False) == 0.0


def test_missing_model_configuration_is_explicit() -> None:
    from invoice_parser_llm import ConfigError, InvoiceParser

    with pytest.raises(ConfigError, match="INVOICE_LLM_MODEL"):
        InvoiceParser(settings=Settings(model=None))
