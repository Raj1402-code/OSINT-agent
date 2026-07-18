"""
agent/search.py
================================================================================
Web search module for the OSINT Investigative Agent.

Uses DuckDuckGo search (via the ddgs package) since it requires no API key.
Pure I/O module — no LLM calls here.
================================================================================
"""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, Field

from agent.utils import get_logger, get_domain, is_valid_url

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """A single web search result."""

    title: str = Field(default="")
    url: str = Field(...)
    snippet: str = Field(default="")
    domain: str = Field(default="")


class WebSearcher:
    """Thin wrapper around DuckDuckGo search with retry and basic cleaning."""

    def __init__(self, max_results: int = 5, max_retries: int = 3) -> None:
        self.max_results = max_results
        self.max_retries = max_retries

    def search(self, query: str) -> List[SearchResult]:
        """Run a web search for a query. Returns [] on repeated failure (never raises)."""
        if not query or not query.strip():
            return []

        DDGS = self._import_ddgs()
        if DDGS is None:
            logger.error(
                "No DuckDuckGo search backend is installed. Run: pip install ddgs"
            )
            return []

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                results: List[SearchResult] = []
                with DDGS() as ddgs:
                    for item in ddgs.text(query, max_results=self.max_results):
                        url = item.get("href") or item.get("url") or ""
                        if not is_valid_url(url):
                            continue
                        results.append(
                            SearchResult(
                                title=(item.get("title") or "").strip(),
                                url=url,
                                snippet=(item.get("body") or "").strip(),
                                domain=get_domain(url),
                            )
                        )
                if not results:
                    logger.warning(
                        f"Search for '{query}' returned 0 results (attempt {attempt}/{self.max_retries}). "
                        "This is usually DuckDuckGo rate-limiting rather than a real lack of results."
                    )
                else:
                    logger.info(f"Search for '{query}' returned {len(results)} result(s).")
                    return results
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(f"Search attempt {attempt}/{self.max_retries} failed for '{query}': {exc}")

            if attempt < self.max_retries:
                time.sleep(2.5 * attempt)

        logger.error(f"Search failed for '{query}' after {self.max_retries} attempts. Last error: {last_exc}")
        return []

    @staticmethod
    def _import_ddgs():
        """Prefer the maintained 'ddgs' package; fall back to the frozen 'duckduckgo_search' name."""
        try:
            from ddgs import DDGS  # type: ignore
            return DDGS
        except ImportError:
            pass
        try:
            from duckduckgo_search import DDGS  # type: ignore
            logger.warning(
                "Using the frozen 'duckduckgo_search' package — it no longer receives "
                "anti-blocking fixes. Run: pip install ddgs (then pip uninstall duckduckgo-search)."
            )
            return DDGS
        except ImportError:
            return None
