"""
agent/utils.py
================================================================================
Shared utilities for the OSINT Investigative Agent: configuration loading,
logging setup, and small text/URL helper functions used across modules.
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import time
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call repeatedly for the same name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
class Settings(BaseModel):
    """
    Strongly-typed, validated application configuration loaded from
    environment variables (populated via a .env file in local/dev use).
    """

    gemini_api_key: str = Field(..., description="Google Gemini API key.")
    gemini_model: str = Field(default="gemini-2.0-flash")
    tavily_api_key: str = Field(default="", description="Tavily search API key.")
    max_search_results_per_subquestion: int = Field(default=5, ge=1, le=20)
    max_subquestions: int = Field(default=5, ge=1, le=10)
    request_timeout_seconds: int = Field(default=10, ge=1, le=60)
    max_page_chars: int = Field(default=8000, ge=500, le=50000)
    scraper_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    )
    log_level: str = Field(default="INFO")

    @field_validator("gemini_api_key")
    @classmethod
    def _validate_api_key(cls, value: str) -> str:
        """Ensure the API key was actually provided and is non-trivial."""
        if not value or value.strip() == "" or "your-gemini-api-key-here" in value:
            raise ValueError(
                "GEMINI_API_KEY is missing or still set to the placeholder "
                "value. Copy .env.example to .env and set a real API key "
                "from https://aistudio.google.com/apikey"
            )
        return value.strip()

    @field_validator("tavily_api_key")
    @classmethod
    def _validate_tavily_key(cls, value: str) -> str:
        """Ensure the Tavily search API key was actually provided."""
        if not value or value.strip() == "" or "your-tavily-api-key-here" in value:
            raise ValueError(
                "TAVILY_API_KEY is missing or still set to the placeholder "
                "value. Get a free key (1,000 searches/month, no credit card) "
                "at https://app.tavily.com and set it in .env / Streamlit Secrets."
            )
        return value.strip()

    @classmethod
    def load(cls) -> "Settings":
        """Build a Settings instance from the current process environment."""
        try:
            return cls(
                gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
                gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
                max_search_results_per_subquestion=int(
                    os.environ.get("MAX_SEARCH_RESULTS_PER_SUBQUESTION", 5)
                ),
                max_subquestions=int(os.environ.get("MAX_SUBQUESTIONS", 5)),
                request_timeout_seconds=int(
                    os.environ.get("REQUEST_TIMEOUT_SECONDS", 10)
                ),
                max_page_chars=int(os.environ.get("MAX_PAGE_CHARS", 8000)),
                scraper_user_agent=os.environ.get(
                    "SCRAPER_USER_AGENT",
                    (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                ),
                log_level=os.environ.get("LOG_LEVEL", "INFO"),
            )
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to load configuration from .env: {exc}") from exc


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def get_domain(url: str) -> str:
    """Extract a clean domain (no www.) from a URL. Returns '' on failure."""
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:  # noqa: BLE001
        return ""


def is_valid_url(url: str) -> bool:
    """Return True if the string looks like a well-formed http(s) URL."""
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:  # noqa: BLE001
        return False


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, cutting on a word boundary where possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated.rstrip() + "…"


def clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines into single spaces/blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def safe_source_id(index: int) -> str:
    """Generate a stable citation id like S1, S2, ... for source numbering."""
    return f"S{index}"


# --------------------------------------------------------------------------
# Gemini rate-limit retry helper
# --------------------------------------------------------------------------
_RETRY_SECONDS_PATTERN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_RETRY_DELAY_SECONDS_PATTERN = re.compile(r"seconds:\s*(\d+)")


def _parse_retry_delay_seconds(error_message: str, default_wait: float) -> float:
    """Pull the server-suggested wait time out of a Gemini 429 error message.

    Gemini's rate-limit errors include a human-readable "Please retry in
    38.17s" plus a structured "retry_delay { seconds: 38 }" block. We check
    both formats and fall back to `default_wait` if neither is found.
    """
    match = _RETRY_SECONDS_PATTERN.search(error_message)
    if match:
        try:
            return float(match.group(1)) + 1.0  # small safety buffer
        except ValueError:
            pass
    match = _RETRY_DELAY_SECONDS_PATTERN.search(error_message)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass
    return default_wait


def call_gemini_with_backoff(
    model,
    content,
    generation_config: dict | None = None,
    max_retries: int = 3,
    default_wait: float = 15.0,
):
    """Call model.generate_content(), automatically waiting and retrying if
    Gemini returns a 429 rate-limit / quota-exceeded error.

    On a 429, Gemini tells us exactly how long to wait before retrying — we
    read that from the error message and sleep for that long (plus a small
    buffer) instead of giving up immediately. Any non-rate-limit error is
    re-raised right away so callers' existing error handling still applies.
    """
    logger = get_logger("agent.retry")
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if generation_config is not None:
                return model.generate_content(content, generation_config=generation_config)
            return model.generate_content(content)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            message = str(exc)
            is_rate_limit = "429" in message or "quota" in message.lower() or "rate limit" in message.lower()

            if not is_rate_limit or attempt == max_retries:
                raise

            wait_seconds = _parse_retry_delay_seconds(message, default_wait)
            logger.warning(
                f"Rate limit hit (attempt {attempt}/{max_retries}); "
                f"waiting {wait_seconds:.1f}s before retrying."
            )
            time.sleep(wait_seconds)

    raise last_exc  # pragma: no cover — unreachable, satisfies type checkers
