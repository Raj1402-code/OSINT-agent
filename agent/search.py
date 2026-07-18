"""
agent/search.py
================================================================================
Web search module for the OSINT Investigative Agent.

Uses the Tavily Search API (https://tavily.com) — a real search API built for
AI agents, with a free tier (1,000 searches/month, no credit card required).

We switched from DuckDuckGo scraping to Tavily because DuckDuckGo's anti-bot
detection frequently serves broken/irrelevant results to requests coming from
datacenter IPs (AWS, GCP, Streamlit Cloud, etc.), even though it usually works
fine from a home network. Tavily is a proper API, so it behaves identically
whether run locally or from any cloud host.

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
    """Thin wrapper around the Tavily Search API with retry and basic cleaning."""

    def __init__(self, api_key: str, max_results: int = 5, max_retries: int = 3) -> None:
        """
        Args:
            api_key: Tavily API key (get a free one at https://app.tavily.com).
            max_results: Max results to request per query.
            max_retries: Retry attempts on transient failures.
        """
        self.api_key = api_key
        self.max_results = max_results
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazily construct the Tavily client. Returns None if the SDK isn't installed
        or no API key was configured."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            logger.error(
                "No TAVILY_API_KEY configured. Get a free key at https://app.tavily.com "
                "and set TAVILY_API_KEY in your .env (or Streamlit Secrets when deployed)."
            )
            return None
        try:
            from tavily import TavilyClient  # type: ignore
        except ImportError:
            logger.error("Tavily SDK is not installed. Run: pip install tavily-python")
            return None
        self._client = TavilyClient(api_key=self.api_key)
        return self._client

    def search(self, query: str) -> List[SearchResult]:
        """Run a web search for a query via Tavily. Returns [] on repeated failure (never raises)."""
        if not query or not query.strip():
            return []

        client = self._get_client()
        if client is None:
            return []

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.search(
                    query=query,
                    max_results=self.max_results,
                    search_depth="basic",
                )
                results: List[SearchResult] = []
                for item in response.get("results", []):
                    url = item.get("url") or ""
                    if not is_valid_url(url):
                        continue
                    results.append(
                        SearchResult(
                            title=(item.get("title") or "").strip(),
                            url=url,
                            snippet=(item.get("content") or "").strip(),
                            domain=get_domain(url),
                        )
                    )

                if not results:
                    logger.warning(
                        f"Search for '{query}' returned 0 results (attempt {attempt}/{self.max_retries})."
                    )
                else:
                    logger.info(f"Search for '{query}' returned {len(results)} result(s).")
                    return results

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(f"Search attempt {attempt}/{self.max_retries} failed for '{query}': {exc}")

            if attempt < self.max_retries:
                time.sleep(2.0 * attempt)

        logger.error(f"Search failed for '{query}' after {self.max_retries} attempts. Last error: {last_exc}")
        return []
