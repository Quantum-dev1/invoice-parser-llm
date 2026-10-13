# Contributing

Thanks for helping improve invoice-parser-llm!

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Checks (all run in CI)

```bash
ruff check .
ruff format --check .
mypy
pytest --cov
```

Tests never call a real LLM: use the scripted `FakeLLM` from `tests/conftest.py`, or the local
stub server in `tests/test_http_e2e.py` for HTTP-level behaviour.

## Guidelines

- Keep the core dependency-light. Provider SDKs are out of scope: anything OpenAI-compatible
  works through `OpenAICompatibleClient`, and other backends can implement the `LLMClient` protocol.
- Validation must stay deterministic. Never ask the model to check arithmetic.
- New validation rules need a test with a concrete document that triggers them.
- Sample data must be clearly fictional. Add new samples to `examples/generate_samples.py` with
  ground truth, never real invoices.
- Never commit API keys or `.env` files.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:` …).

## Reporting extraction problems

Please do not upload real invoices to public issues. Describe the layout, or build a synthetic
reproduction with `examples/generate_samples.py`.
