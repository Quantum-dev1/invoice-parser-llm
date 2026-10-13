"""Configuration from environment variables. Nothing is hardcoded: endpoint, model and key are yours."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _first(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """All tunables. Build with :meth:`from_env` or construct directly."""

    model: str | None = None
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = field(default=None, repr=False)
    vision_model: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    json_mode: bool = True
    temperature: float = 0.0
    max_attempts: int = 3
    tolerance: Decimal = Decimal("0.01")
    max_pages: int = 5
    min_text_chars: int = 40
    image_dpi: int = 150
    max_text_chars: int = 60_000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``INVOICE_LLM_*`` / ``INVOICE_*`` variables (with ``OPENAI_*`` fallbacks)."""
        e = os.environ if env is None else env
        tolerance_raw = _first(e, "INVOICE_TOLERANCE") or "0.01"
        try:
            tolerance = Decimal(tolerance_raw)
        except Exception as exc:
            raise ConfigError(f"INVOICE_TOLERANCE must be a number, got {tolerance_raw!r}") from exc
        return cls(
            model=_first(e, "INVOICE_LLM_MODEL"),
            base_url=_first(e, "INVOICE_LLM_BASE_URL", "OPENAI_BASE_URL") or cls.base_url,
            api_key=_first(e, "INVOICE_LLM_API_KEY", "OPENAI_API_KEY"),
            vision_model=_first(e, "INVOICE_LLM_VISION_MODEL"),
            timeout=_float(e, "INVOICE_LLM_TIMEOUT", cls.timeout),
            max_retries=_int(e, "INVOICE_LLM_MAX_RETRIES", cls.max_retries),
            json_mode=_bool(_first(e, "INVOICE_LLM_JSON_MODE"), cls.json_mode),
            temperature=_float(e, "INVOICE_LLM_TEMPERATURE", cls.temperature),
            max_attempts=_int(e, "INVOICE_MAX_ATTEMPTS", cls.max_attempts),
            tolerance=tolerance,
            max_pages=_int(e, "INVOICE_MAX_PAGES", cls.max_pages),
            min_text_chars=_int(e, "INVOICE_MIN_TEXT_CHARS", cls.min_text_chars),
            image_dpi=_int(e, "INVOICE_IMAGE_DPI", cls.image_dpi),
        )

    def with_overrides(self, **changes: object) -> Settings:
        """Return a copy with non-``None`` overrides applied (handy for CLI flags)."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})  # type: ignore[arg-type]

    def require_model(self) -> str:
        if not self.model:
            raise ConfigError(
                "No model configured. Set INVOICE_LLM_MODEL (and INVOICE_LLM_BASE_URL / INVOICE_LLM_API_KEY "
                "for your provider), e.g. INVOICE_LLM_MODEL=gpt-4o-mini or a local Ollama model."
            )
        return self.model
