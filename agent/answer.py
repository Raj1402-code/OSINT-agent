"""
agent/answer.py
================================================================================
Final answer synthesis module for the OSINT Investigative Agent.

Takes the full evidence pool plus the verification result (conflicts +
confidence) and asks Gemini to write a final, cited answer to the user's
original question, with a numbered source list ([S1], [S2], ...).
================================================================================
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.extractor import EvidenceItem
from agent.verifier import VerificationResult, ConfidenceLevel
from agent.utils import get_logger, safe_source_id

logger = get_logger(__name__)


class SourceEntry(BaseModel):
    """A single numbered source in the final answer's reference list."""

    citation_id: str = Field(...)
    url: str = Field(...)
    title: str = Field(default="")
    domain: str = Field(default="")
    why_used: List[str] = Field(default_factory=list)


class FinalAnswer(BaseModel):
    """The complete synthesized answer returned to the user."""

    answer_text: str = Field(default="")
    sources: List[SourceEntry] = Field(default_factory=list)
    confidence_level: ConfidenceLevel = Field(default="Medium")
    confidence_rationale: str = Field(default="")
    conflicts_summary: List[str] = Field(default_factory=list)


class AnswerGenerator:
    """Uses Gemini to synthesize the final, cited answer."""

    SYSTEM_PROMPT = """You are a careful OSINT investigative analyst writing a final report.

You will be given the user's original investigative question, a pool of
evidence items (each tagged with a citation id like [S1], [S2], ...), and a
verification summary noting any conflicts between sources plus an overall
confidence level.

YOUR TASK: Write a clear, well-organized, neutral answer to the original
question, using ONLY the evidence provided.

STRICT RULES:
- Every substantive claim in your answer MUST be followed by the citation
  id(s) of the evidence it's based on, in square brackets, e.g. "Company X
  announced the merger in March 2024 [S2][S5]."
- NEVER state a fact that isn't backed by at least one evidence item.
- If sources conflict on a point, present both sides and cite each side's
  sources, rather than picking one as correct.
- Do not add outside knowledge not present in the evidence pool.
- Write in clear, neutral, journalistic prose — no editorializing.
- Organize the answer with short paragraphs or a few subheadings if it
  covers multiple angles.
- Also produce a short list of plain-language conflict summaries (empty
  list if there were no conflicts) for display separately from the answer.

Output ONLY valid JSON matching this schema, nothing else:
{
  "answer_text": "string - full answer with inline [S#] citations",
  "conflicts_summary": ["string"]
}
"""

    def __init__(self, model: str) -> None:
        self.model_name = model

    def generate(
        self,
        original_question: str,
        evidence_items: List[EvidenceItem],
        verification: VerificationResult,
    ) -> FinalAnswer:
        """Generate the final cited answer. Never raises."""
        if not evidence_items:
            logger.warning("No evidence available; returning fallback answer.")
            return FinalAnswer(
                answer_text=(
                    "I was unable to gather sufficient evidence from web sources to answer this "
                    "question. This may be due to search or scraping failures, or because the topic "
                    "has very little coverage online. Please try rephrasing the question or narrowing "
                    "its scope."
                ),
                sources=[],
                confidence_level="Low",
                confidence_rationale=verification.confidence_rationale,
                conflicts_summary=[],
            )

        source_entries, url_to_citation_id = self._build_source_registry(evidence_items)
        evidence_payload = self._format_evidence_for_prompt(
            original_question, evidence_items, verification, url_to_citation_id
        )

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_PROMPT,
            )
            response = model.generate_content(
                evidence_payload,
                generation_config={"response_mime_type": "application/json"},
            )
            raw_text = (response.text or "").strip()
            parsed = self._parse_json_response(raw_text)

            answer_text = (parsed.get("answer_text") or "").strip()
            conflicts_summary = [c.strip() for c in parsed.get("conflicts_summary", []) if c and c.strip()]

            if not answer_text:
                logger.warning("Answer model returned empty answer_text; using fallback message.")
                answer_text = (
                    "The investigation gathered evidence, but the answer-generation step failed to "
                    "produce a valid response. Please review the raw evidence and sources below."
                )

            self._attach_why_used(source_entries, evidence_items)

            logger.info(
                f"Final answer generated with {len(source_entries)} source(s), "
                f"{len(conflicts_summary)} conflict summary item(s)."
            )

            return FinalAnswer(
                answer_text=answer_text,
                sources=source_entries,
                confidence_level=verification.confidence_level,
                confidence_rationale=verification.confidence_rationale,
                conflicts_summary=conflicts_summary,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Answer generation failed: {exc}")
            return FinalAnswer(
                answer_text=(
                    "An error occurred while generating the final answer. The investigation did "
                    "gather evidence from the sources listed below — please review them directly."
                ),
                sources=source_entries,
                confidence_level=verification.confidence_level,
                confidence_rationale=verification.confidence_rationale,
                conflicts_summary=[],
            )

    @staticmethod
    def _build_source_registry(
        evidence_items: List[EvidenceItem],
    ) -> Tuple[List[SourceEntry], Dict[str, str]]:
        """Assign a stable citation id (S1, S2, ...) to each unique source URL."""
        source_entries: List[SourceEntry] = []
        url_to_citation_id: Dict[str, str] = {}

        for item in evidence_items:
            if item.source_url in url_to_citation_id:
                continue
            citation_id = safe_source_id(len(source_entries) + 1)
            url_to_citation_id[item.source_url] = citation_id
            source_entries.append(
                SourceEntry(
                    citation_id=citation_id,
                    url=item.source_url,
                    title=item.source_title,
                    domain=item.source_domain,
                    why_used=[],
                )
            )

        return source_entries, url_to_citation_id

    @staticmethod
    def _attach_why_used(source_entries: List[SourceEntry], evidence_items: List[EvidenceItem]) -> None:
        """Populate each source's why_used list with the claims drawn from it."""
        by_url: Dict[str, SourceEntry] = {s.url: s for s in source_entries}
        for item in evidence_items:
            entry = by_url.get(item.source_url)
            if entry is not None and item.claim not in entry.why_used:
                entry.why_used.append(item.claim)

    @staticmethod
    def _format_evidence_for_prompt(
        original_question: str,
        evidence_items: List[EvidenceItem],
        verification: VerificationResult,
        url_to_citation_id: Dict[str, str],
    ) -> str:
        lines = [f"ORIGINAL QUESTION:\n{original_question}\n", "EVIDENCE POOL:\n"]
        for item in evidence_items:
            citation_id = url_to_citation_id.get(item.source_url, "S?")
            lines.append(
                f"[{citation_id}] Claim: {item.claim}\n"
                f"    Quote: \"{item.supporting_quote}\"\n"
                f"    Source: {item.source_domain} ({item.source_url})\n"
            )

        lines.append("\nVERIFICATION SUMMARY:")
        lines.append(f"Confidence level: {verification.confidence_level}")
        lines.append(f"Confidence rationale: {verification.confidence_rationale}")
        if verification.conflicts:
            lines.append("Detected conflicts:")
            for conflict in verification.conflicts:
                lines.append(f"- {conflict.topic}: {conflict.description}")
        else:
            lines.append("Detected conflicts: none")

        return "\n".join(lines)

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
            if isinstance(parsed, dict):
                return parsed
            logger.warning(f"Unexpected JSON shape from answer model: {cleaned[:200]}")
            return {}
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON from answer model: {exc}. Raw: {cleaned[:200]}")
            return {}
