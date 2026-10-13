"""Render docs/cli-parse.svg — the real CLI summary output for a scripted self-correction run.

The LLM is replaced by two scripted replies (no network): the first misreads one line amount
(617.50 → 671.50), arithmetic validation catches it, and the second reply is correct. Everything
else — PDF extraction, validation, retry, confidence and the summary table — is the real code path.

    python scripts/render_cli_screenshot.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from rich.console import Console

from invoice_parser_llm import InvoiceParser, LLMResponse, Settings, cli
from invoice_parser_llm.llm import Message

ROOT = Path(__file__).resolve().parents[1]
truth = json.loads((ROOT / "examples/expected/acme-cloud-invoice.json").read_text())
misread = copy.deepcopy(truth)
misread["line_items"][2]["amount"] = 671.50
misread["subtotal"] = 1598.50


class Scripted:
    model = "your-model"

    def __init__(self) -> None:
        self.replies = [misread, truth]

    def complete(self, messages: list[Message]) -> LLMResponse:
        return LLMResponse(json.dumps(self.replies.pop(0)), self.model)


console = Console(record=True, width=100, force_terminal=True, color_system="truecolor")
cli.err = console
parser = InvoiceParser(client=Scripted(), settings=Settings(model="your-model"))
result = parser.parse(ROOT / "examples/invoices/acme-cloud-invoice.pdf")

console.print("[bold green]$[/] invoice-parser parse examples/invoices/acme-cloud-invoice.pdf -o acme.json")
console.print(
    "[dim]# attempt 1: line 3 read as 671.50 → line_amount_mismatch, subtotal_mismatch → correction request[/]"
)
console.print("[dim]# attempt 2: corrected → all checks pass[/]")
cli._summary(result)
out = ROOT / "docs" / "cli-parse.svg"
out.parent.mkdir(exist_ok=True)
console.save_svg(str(out), title="invoice-parser")
print(f"wrote {out.relative_to(ROOT)}")
