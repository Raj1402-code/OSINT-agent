"""
agent/extractor.py
================================================================================
Page scraping and evidence extraction for the OSINT Investigative Agent.

PageScraper: fetches a URL and extracts clean article text (no LLM involved).
EvidenceExtractor: uses Gemini to pull discrete, atomic factual claims out of
that scraped text, each grounded in a supporting quote from the page.
================================================================================
"""

from __future__ import annotations

import json
from typing import List, Optional

import requests
import trafilatura
from bs4 import BeautifulSoup
import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.rate_limit import call_with_rate_limit_retry
from agent.search import SearchResult
from agent.utils import get_logger, get_domain, truncate_text, clean_whitespace

logger = get_logger(__name__)


class ScrapedPage(BaseModel):
    """The cleaned text content of a fetched web page."""

    url: str = Field(...)
    title: str = Field(default="")
    text: str = Field(default="")
    domain: str = Field(default="")
    success: bool = Field(default=False)


class EvidenceItem(BaseModel):
    """A single atomic factual claim grounded in a source page."""

    claim: str = Field(...)
    supporting_quote: str = Field(default="")
    source_url: str = Field(...)
    source_title: str = Field(default="")
    source_domain: str = Field(default="")
    sub_question: str = Field(default="")


class PageScraper:
    """Fetches URLs and extracts clean readable text using trafilatura, with a
    BeautifulSoup fallback for pages trafilatura can't parse well."""

    def __init__(self, user_agent: str, timeout_seconds: int = 10, max_page_chars: int = 8000) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_page_chars = max_page_chars

    def scrape(self, result: SearchResult) -> ScrapedPage:
        """Fetch and clean a page's text. Never raises — returns success=False on failure."""
        url = result.url
        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            html = response.text

            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            title = self._extract_title(html) or result.title

            if not text or len(text.strip()) < 100:
                text = self._fallback_extract(html)

            text = clean_whitespace(text)
            if not text:
                logger.warning(f"No extractable text found for {url}")
                return ScrapedPage(url=url, title=title, text="", domain=get_domain(url), success=False)

            text = truncate_text(text, self.max_page_chars)
            return ScrapedPage(url=url, title=title, text=text, domain=get_domain(url), success=True)

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to scrape {url}: {exc}")
            return ScrapedPage(url=url, title=result.title, text="", domain=get_domain(url), success=False)

    @staticmethod
    def _extract_title(html: str) -> Optional[str]:
        try:
            soup = BeautifulSoup(html, "lxml")
            if soup.title and soup.title.string:
                return soup.title.string.strip()
        except Exception:  # noqa: BLE001
            pass
        return None

    @staticmethod
    def _fallback_extract(html: str) -> str:
        """Basic BeautifulSoup fallback: grab all paragraph text."""
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            return "\n\n".join(p for p in paragraphs if len(p) > 40)
        except Exception:  # noqa: BLE001
            return ""


class EvidenceExtractor:
    """Uses Gemini to pull discrete, atomic factual claims out of scraped page text."""

    SYSTEM_PROMPT = """You are a meticulous OSINT evidence-extraction assistant.

You will be given:
1. An investigative sub-question.
2. The cleaned text content of a single web page.

Extract ONLY factual claims that are EXPLICITLY STATED in the provided page
text and RELEVANT to the sub-question.

STRICT RULES:
- NEVER invent, infer beyond the text, assume, or add any fact not explicitly
  present in the page text.
- If the text is ambiguous, extract it as stated — do not resolve ambiguity yourself.
- If the page contains NOTHING relevant to the sub-question, return an empty
  "evidence" list.
- Each claim must be atomic: one discrete fact per item.
- Include a short supporting_quote copied verbatim (or near-verbatim) from
  the page text for each claim.
- No opinions, analysis, or commentary — only what the source text states.
- Output ONLY valid JSON matching this schema, nothing else:

{
  "evidence": [
    {"claim": "string", "supporting_quote": "string"}
  ]
}
"""

    def __init__(self, model: str) -> None:
        """
        Args:
            model: Gemini model name, e.g. "gemini-2.0-flash".
        """
        self.model_name = model

    def extract(self, page: ScrapedPage, sub_question: str) -> List[EvidenceItem]:
        """Extract evidence items from a scraped page. Never raises."""
        if not page.success or not page.text:
            return []

        user_message = (
            f"SUB-QUESTION:\n{sub_question}\n\n"
            f"PAGE TITLE: {page.title}\n"
            f"PAGE URL: {page.url}\n\n"
            f"PAGE TEXT:\n{page.text}"
        )

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_PROMPT,
            )

            def _call():
                return model.generate_content(
                    user_message,
                    generation_config={"response_mime_type": "application/json"},
                )

            response = call_with_rate_limit_retry(_call, context=f"extractor:{page.url}")
            raw_text = (response.text or "").strip()
            parsed = self._parse_json_response(raw_text)

            evidence_items: List[EvidenceItem] = []
            for item in parsed.get("evidence", []):
                claim = (item.get("claim") or "").strip()
                quote = (item.get("supporting_quote") or "").strip()
                if not claim:
                    continue
                evidence_items.append(
                    EvidenceItem(
                        claim=claim,
                        supporting_quote=quote,
                        source_url=page.url,
                        source_title=page.title,
                        source_domain=page.domain,
                        sub_question=sub_question,
                    )
                )

            logger.info(f"Extracted {len(evidence_items)} evidence item(s) from {page.url}")
            return evidence_items

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Evidence extraction failed for {page.url}: {exc}")
            return []

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "evidence" in parsed:
                return parsed
            logger.warning(f"Unexpected JSON shape from extraction model: {cleaned[:200]}")
            return {"evidence": []}
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON from extraction model: {exc}. Raw: {cleaned[:200]}")
            return {"evidence": []}
