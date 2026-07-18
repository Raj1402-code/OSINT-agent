"""
agent/verifier.py
================================================================================
Evidence verification module for the OSINT Investigative Agent.

Takes the full pool of evidence gathered across all sub-questions and asks
Gemini to cross-check it for contradictions/conflicts between sources, and
to assign an overall confidence level to the eventual answer.
================================================================================
"""

from __future__ import annotations

import json
from typing import List, Literal

import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.extractor import EvidenceItem
from agent.utils import get_logger

logger = get_logger(__name__)

ConfidenceLevel = Literal["High", "Medium", "Low"]


class ConflictReport(BaseModel):
    """A detected contradiction between two or more pieces of evidence."""

    topic: str = Field(default="")
    description: str = Field(default="")
    conflicting_claims: List[str] = Field(default_factory=list)
    conflicting_sources: List[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Output of the verification step: conflicts found + overall confidence."""

    conflicts: List[ConflictReport] = Field(default_factory=list)
    confidence_level: ConfidenceLevel = Field(default="Medium")
    confidence_rationale: str = Field(default="")


class EvidenceVerifier:
    """Uses Gemini to cross-check the evidence pool for conflicts and assign confidence."""

    SYSTEM_PROMPT = """You are a rigorous OSINT fact-verification analyst.

You will be given a pool of evidence items gathered from multiple web
sources during an investigation, each with a claim, a supporting quote,
and a source URL/domain.

YOUR TASKS:
1. Identify any CONFLICTS: cases where two or more sources make claims that
   contradict each other on the same topic or fact. Only flag genuine
   contradictions, not simply different emphasis or incomplete overlap.
2. Assess an OVERALL CONFIDENCE LEVEL for an answer built from this evidence:
   - "High": Multiple independent, credible sources agree; little or no conflict.
   - "Medium": Reasonable evidence exists but is thin, from few sources, or
     has some unresolved conflicts.
   - "Low": Evidence is sparse, comes from very few or low-quality sources,
     or has significant unresolved conflicts.
3. Give a short rationale for the confidence level you assign.

STRICT RULES:
- Base your analysis ONLY on the evidence items provided — do not use outside
  knowledge to judge truth, only to detect internal contradictions.
- Do not fabricate conflicts that aren't actually present in the evidence.

Output ONLY valid JSON matching this schema, nothing else:
{
  "conflicts": [
    {"topic": "string", "description": "string",
     "conflicting_claims": ["string"], "conflicting_sources": ["url"]}
  ],
  "confidence_level": "High" | "Medium" | "Low",
  "confidence_rationale": "string"
}
"""

    def __init__(self, model: str) -> None:
        self.model_name = model

    def verify(self, evidence_items: List[EvidenceItem]) -> VerificationResult:
        """Analyze evidence for conflicts and assign confidence. Never raises."""
        if not evidence_items:
            logger.warning("No evidence items to verify; returning Low confidence.")
            return VerificationResult(
                conflicts=[],
                confidence_level="Low",
                confidence_rationale="No evidence was successfully gathered, so confidence cannot be established.",
            )

        evidence_payload = self._format_evidence_for_prompt(evidence_items)

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

            conflicts = self._build_conflicts(parsed)
            confidence_level = self._validate_confidence_level(parsed.get("confidence_level"))
            confidence_rationale = (parsed.get("confidence_rationale") or "").strip()

            logger.info(f"Verification complete: {len(conflicts)} conflict(s), confidence={confidence_level}")

            return VerificationResult(
                conflicts=conflicts,
                confidence_level=confidence_level,
                confidence_rationale=confidence_rationale or "No rationale provided by verification model.",
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Verification failed: {exc}. Falling back to Medium confidence.")
            return VerificationResult(
                conflicts=[],
                confidence_level="Medium",
                confidence_rationale=(
                    "Automated conflict-detection failed to run; confidence defaulted to Medium. "
                    "Review sources manually for potential contradictions."
                ),
            )

    @staticmethod
    def _format_evidence_for_prompt(evidence_items: List[EvidenceItem]) -> str:
        lines = ["EVIDENCE POOL:\n"]
        for idx, item in enumerate(evidence_items, start=1):
            lines.append(
                f"[{idx}] Claim: {item.claim}\n"
                f"    Quote: \"{item.supporting_quote}\"\n"
                f"    Source: {item.source_domain} ({item.source_url})\n"
                f"    Related sub-question: {item.sub_question}\n"
            )
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
            logger.warning(f"Unexpected JSON shape from verification model: {cleaned[:200]}")
            return {}
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON from verification model: {exc}. Raw: {cleaned[:200]}")
            return {}

    @staticmethod
    def _build_conflicts(parsed: dict) -> List[ConflictReport]:
        conflicts: List[ConflictReport] = []
        for item in parsed.get("conflicts", []):
            try:
                conflicts.append(
                    ConflictReport(
                        topic=(item.get("topic") or "").strip(),
                        description=(item.get("description") or "").strip(),
                        conflicting_claims=[c for c in item.get("conflicting_claims", []) if c],
                        conflicting_sources=[s for s in item.get("conflicting_sources", []) if s],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Skipping malformed conflict {item!r}: {exc}")
                continue
        return conflicts

    @staticmethod
    def _validate_confidence_level(value: object) -> ConfidenceLevel:
        if value in ("High", "Medium", "Low"):
            return value  # type: ignore[return-value]
        return "Medium"
