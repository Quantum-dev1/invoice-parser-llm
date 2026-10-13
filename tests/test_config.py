from __future__ import annotations

from decimal import Decimal

import pytest

from invoice_parser_llm import ConfigError, Settings


def test_defaults_without_env() -> None:
    s = Settings.from_env({})
    assert s.model is None
    assert s.base_url == "https://api.openai.com/v1"
    assert s.api_key is None
    assert s.json_mode is True
    assert s.tolerance == Decimal("0.01")


def test_reads_invoice_vars_with_openai_fallbacks() -> None:
    s = Settings.from_env(
        {
            "INVOICE_LLM_MODEL": "llama3.2-vision",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "OPENAI_API_KEY": "fallback",
            "INVOICE_LLM_JSON_MODE": "false",
            "INVOICE_MAX_ATTEMPTS": "5",
            "INVOICE_TOLERANCE": "0.05",
            "INVOICE_LLM_TIMEOUT": "30",
        }
    )
    assert s.model == "llama3.2-vision"
    assert s.base_url == "http://localhost:11434/v1"
    assert s.api_key == "fallback"
    assert s.json_mode is False
    assert s.max_attempts == 5
    assert s.tolerance == Decimal("0.05")
    assert s.timeout == 30.0
    # Dedicated variables win over OPENAI_* fallbacks.
    assert Settings.from_env({"INVOICE_LLM_API_KEY": "own", "OPENAI_API_KEY": "other"}).api_key == "own"


def test_api_key_is_not_in_repr() -> None:
    assert "supersecret" not in repr(Settings(model="m", api_key="supersecret"))


@pytest.mark.parametrize(
    "env",
    [{"INVOICE_MAX_ATTEMPTS": "three"}, {"INVOICE_LLM_TIMEOUT": "soon"}, {"INVOICE_TOLERANCE": "tiny"}],
)
def test_invalid_numbers_raise(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_overrides_ignore_none() -> None:
    s = Settings(model="a").with_overrides(model=None, base_url="http://x/v1")
    assert (s.model, s.base_url) == ("a", "http://x/v1")
