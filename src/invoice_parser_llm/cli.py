"""Command-line interface: ``invoice-parser parse | batch | validate | schema | serve``."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import ConfigError, Settings
from .documents import UnsupportedDocumentError
from .llm import LLMError
from .parser import InvoiceParser, ParseResult
from .schema import Invoice, invoice_json_schema
from .validation import validate_invoice

app = typer.Typer(
    name="invoice-parser",
    help="Turn invoice and receipt PDFs/images into validated JSON with any OpenAI-compatible LLM.",
    no_args_is_help=True,
    add_completion=False,
)
err = Console(stderr=True)


def make_parser(
    model: str | None = None, base_url: str | None = None, vision_model: str | None = None
) -> InvoiceParser:
    """Factory used by the CLI (patched in tests)."""
    settings = Settings.from_env().with_overrides(model=model, base_url=base_url, vision_model=vision_model)
    return InvoiceParser(settings=settings)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"invoice-parser-llm {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version and exit.")
    ] = False,
) -> None:
    """invoice-parser-llm."""


def _summary(result: ParseResult) -> None:
    inv = result.invoice
    status = "[bold red]needs review[/]" if result.needs_review else "[bold green]ok[/]"
    err.print(
        f"[bold]{result.file}[/]  {status}  ·  confidence [bold]{result.confidence:.2f}[/]  ·  "
        f"{result.extraction}  ·  {result.attempts} attempt(s)"
    )
    table = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    table.add_column(style="dim")
    table.add_column()
    if inv:

        def money(value: Decimal | None) -> str:
            return "—" if value is None else f"{value:,.2f} {inv.currency or ''}".strip()

        table.add_row("Vendor", inv.vendor.name or "—")
        table.add_row("Buyer", (inv.buyer.name if inv.buyer else None) or "—")
        table.add_row("Number / date", f"{inv.invoice_number or '—'} / {inv.issue_date or '—'}")
        table.add_row("Line items", str(len(inv.line_items)))
        table.add_row("Subtotal", money(inv.subtotal))
        table.add_row("Tax", money(inv.tax_total))
        table.add_row("Total", f"[bold]{money(inv.total)}[/]")
    for issue in result.issues:
        color = "red" if issue.severity == "error" else "yellow"
        table.add_row(f"[{color}]{issue.code}[/{color}]", issue.message)
    err.print(table)


def _fail(message: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {message}")
    return typer.Exit(code)


@app.command()
def parse(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="PDF or image file.")],
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write JSON here instead of stdout.")] = None,
    model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_MODEL.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_BASE_URL.")] = None,
    vision_model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_VISION_MODEL.")] = None,
    invoice_only: Annotated[bool, typer.Option("--invoice-only", help="Output only the invoice object.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Do not print the summary table.")] = False,
    strict: Annotated[bool, typer.Option(help="Exit with code 2 when the result needs review.")] = False,
) -> None:
    """Parse one invoice/receipt and print JSON."""
    try:
        parser = make_parser(model, base_url, vision_model)
        result = parser.parse(file)
    except (ConfigError, UnsupportedDocumentError, LLMError) as exc:
        raise _fail(str(exc)) from exc

    payload = (
        (result.invoice.model_dump(mode="json") if result.invoice else None)
        if invoice_only
        else result.model_dump(mode="json")
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        typer.echo(text)
    if not quiet:
        _summary(result)
    if strict and result.needs_review:
        raise typer.Exit(2)


@app.command()
def batch(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder with PDFs/images.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="JSONL output file.")] = Path("results.jsonl"),
    csv_path: Annotated[Path | None, typer.Option("--csv", help="Also write a one-row-per-document CSV.")] = None,
    items_csv: Annotated[
        Path | None, typer.Option("--items-csv", help="Also write a one-row-per-line-item CSV.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, max=64, help="Parallel requests.")] = 4,
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Include subfolders.")] = False,
    model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_MODEL.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_BASE_URL.")] = None,
    vision_model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_VISION_MODEL.")] = None,
) -> None:
    """Parse every supported file in a folder → JSONL (+ optional CSVs)."""
    from .batch import find_documents, parse_many, write_items_csv, write_jsonl, write_summary_csv

    paths = find_documents(folder, recursive=recursive)
    if not paths:
        raise _fail(f"no supported documents found in {folder}")
    try:
        parser = make_parser(model, base_url, vision_model)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    items = []
    with err.status(f"Parsing {len(paths)} document(s)…") as status:
        for item in parse_many(parser, paths, workers=workers):
            items.append(item)
            status.update(f"Parsed {len(items)}/{len(paths)}: {item.path.name}")
    items.sort(key=lambda i: str(i.path))

    with output.open("w", encoding="utf-8") as fp:
        write_jsonl(items, fp)
    if csv_path:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            write_summary_csv(items, fp)
    if items_csv:
        with items_csv.open("w", encoding="utf-8", newline="") as fp:
            write_items_csv(items, fp)

    ok = sum(1 for i in items if i.ok and i.result and not i.result.needs_review)
    review = sum(1 for i in items if i.ok and i.result and i.result.needs_review)
    failed = len(items) - ok - review
    err.print(f"[green]{ok} ok[/green], [yellow]{review} need review[/yellow], [red]{failed} failed[/red] → {output}")


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Invoice JSON (or a parse result).")],
    tolerance: Annotated[float, typer.Option(help="Allowed difference per summed term.")] = 0.01,
) -> None:
    """Run the arithmetic checks on an existing invoice JSON — no LLM needed."""
    data = json.loads(file.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("invoice"), dict):
        data = data["invoice"]
    invoice = Invoice.model_validate(data)
    issues = validate_invoice(invoice, str(tolerance))
    typer.echo(json.dumps([i.model_dump(mode="json") for i in issues], indent=2, ensure_ascii=False))
    if any(i.severity == "error" for i in issues):
        raise typer.Exit(2)


@app.command(name="eval")
def evaluate(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder with documents.")],
    truth: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder with ground-truth JSON files.")],
    workers: Annotated[int, typer.Option(min=1, max=64, help="Parallel requests.")] = 4,
    report: Annotated[Path | None, typer.Option("--report", help="Write the full report as JSON.")] = None,
    model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_MODEL.")] = None,
    base_url: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_BASE_URL.")] = None,
    vision_model: Annotated[str | None, typer.Option(help="Override INVOICE_LLM_VISION_MODEL.")] = None,
) -> None:
    """Measure field-level accuracy against labeled ground truth (matched by file name)."""
    from .batch import find_documents, parse_many
    from .evaluation import build_report, match_truth, score_document

    truth_files = sorted(truth.glob("*.json"))
    pairs = {doc: match_truth(doc, truth_files) for doc in find_documents(folder)}
    paths = [doc for doc, t in pairs.items() if t is not None]
    if not paths:
        raise _fail("no documents with matching ground truth found")
    try:
        parser = make_parser(model, base_url, vision_model)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc

    scores = []
    with err.status(f"Evaluating {len(paths)} document(s)…"):
        for item in parse_many(parser, paths, workers=workers):
            truth_path = pairs[item.path]
            assert truth_path is not None
            predicted = item.result.invoice if item.result else None
            scores.append(
                score_document(item.path.name, truth_path, predicted, item.result.needs_review if item.result else None)
            )
    scores.sort(key=lambda s: s.file)
    result = build_report(scores)

    docs = Table(title="Documents", title_justify="left")
    for col in ("File", "Accuracy", "Needs review", "Wrong fields"):
        docs.add_column(col)
    for s in result.documents:
        wrong = ", ".join(k for k, ok in s.fields.items() if not ok) or "—"
        docs.add_row(s.file, f"{s.accuracy:.0%}", str(s.needs_review), wrong)
    err.print(docs)
    fields = Table(title="Field accuracy", title_justify="left")
    fields.add_column("Field")
    fields.add_column("Accuracy", justify="right")
    for name, acc in result.field_accuracy.items():
        color = "green" if acc == 1 else "yellow" if acc >= 0.8 else "red"
        fields.add_row(name, f"[{color}]{acc:.0%}[/{color}]")
    err.print(fields)
    err.print(
        f"Overall field accuracy: [bold]{result.overall_accuracy:.1%}[/bold] · "
        f"exact documents: {result.exact_documents}/{len(result.documents)}"
    )
    if report:
        report.write_text(result.model_dump_json(indent=2), encoding="utf-8")


@app.command()
def schema() -> None:
    """Print the JSON Schema of the invoice object."""
    typer.echo(json.dumps(invoice_json_schema(), indent=2))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Run the HTTP API (needs: pip install "invoice-parser-llm\\[api]")."""
    try:
        import uvicorn
    except ImportError as exc:
        raise _fail('the API needs extra dependencies: pip install "invoice-parser-llm[api]"') from exc
    uvicorn.run("invoice_parser_llm.api:app", host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
