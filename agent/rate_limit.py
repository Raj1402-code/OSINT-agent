"""
agent/rate_limit.py
================================================================================
Shared retry helper for Gemini API calls.

The free tier of gemini-3.1-flash-lite allows only 15 requests/minute.
This pipeline can easily make 20-30+ calls in a single investigation
(1 planner call + 1 per extracted source + 1 verify + 1 answer), so any
individual call can hit a 429 even though the overall workload is legitimate.

Google's 429 response includes a `retry_delay` telling us exactly how long
to wait. Instead of failing immediately, we catch 429s, sleep for the
suggested delay (plus a small buffer), and retry — up to a max number of
attempts — before giving up.
================================================================================
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, TypeVar

from agent.utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)")

# Free-tier gemini-3.1-flash-lite allows 15 requests/minute PER PROJECT, shared
# across every call site (planner, extractor, verifier, answer). A single
# investigation can make one extraction call per scraped source, so without
# pacing we blow through 15/min well before the run finishes.
#
# 60s / 15 = 4.0s minimum spacing. We pad it a bit for safety margin.
_MIN_CALL_INTERVAL_SECONDS = 4.3
_last_call_lock = threading.Lock()
_last_call_ts = 0.0


def throttle() -> None:
    """Block until enough time has passed since the last Gemini call to stay under quota.

    Call this immediately before every genai.GenerativeModel(...).generate_content()
    call, across all modules (planner, extractor, verifier, answer) — they all
    share the same per-project-per-model quota bucket.
    """
    global _last_call_ts
    with _last_call_lock:
        now = time.monotonic()
        elapsed = now - _last_call_ts
        if elapsed < _MIN_CALL_INTERVAL_SECONDS:
            wait = _MIN_CALL_INTERVAL_SECONDS - elapsed
            logger.info(f"Pacing Gemini call: waiting {wait:.1f}s to stay under rate limit.")
            time.sleep(wait)
        _last_call_ts = time.monotonic()


def _extract_retry_delay(exc: Exception, default: float = 20.0) -> float:
    """Pull the server-suggested retry delay (in seconds) out of a 429 error message."""
    match = _RETRY_DELAY_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 2.0  # small buffer on top of Google's suggestion
    return default


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "quota" in text.lower() or "ResourceExhausted" in type(exc).__name__


def call_with_rate_limit_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    context: str = "",
) -> T:
    """
    Call fn(), retrying with the server-suggested delay if it fails with a 429.

    Re-raises the last exception if all attempts are exhausted, or if the
    error isn't a rate-limit error (no point retrying those).
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        throttle()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _is_rate_limit_error(exc):
                raise  # not a rate limit issue, don't retry

            if attempt == max_attempts:
                break

            delay = _extract_retry_delay(exc)
            logger.warning(
                f"Rate limit hit{f' ({context})' if context else ''}, "
                f"attempt {attempt}/{max_attempts}. Waiting {delay:.0f}s before retry."
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc
