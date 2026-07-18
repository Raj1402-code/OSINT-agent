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

    @classmethod
    def load(cls) -> "Settings":
        """Build a Settings instance from the current process environment."""
        try:
            return cls(
                gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
                gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
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
