from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from invoice_parser_llm import InvoiceParser, LLMResponse, Settings
from invoice_parser_llm.llm import Message

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "invoices"
EXPECTED = ROOT / "examples" / "expected"


class FakeLLM:
    """Scripted stand-in for an LLM. Each call pops the next reply (str, dict → JSON, or Exception)."""

    def __init__(self, *replies: str | dict[str, Any] | Exception, model: str = "fake-model") -> None:
        self.replies = list(replies)
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(content=content, model=self.model, usage={"prompt_tokens": 100, "completion_tokens": 50})


def load_expected(name: str) -> dict[str, Any]:
    return json.loads((EXPECTED / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def settings() -> Settings:
    return Settings(model="fake-model", max_attempts=3)


@pytest.fixture
def acme_expected() -> dict[str, Any]:
    return load_expected("acme-cloud-invoice")


@pytest.fixture
def initech_expected() -> dict[str, Any]:
    return load_expected("initech-office-supplies")


@pytest.fixture
def northwind_expected() -> dict[str, Any]:
    return load_expected("northwind-receipt")


def make_parser(fake: FakeLLM, settings: Settings) -> InvoiceParser:
    return InvoiceParser(client=fake, settings=settings)
