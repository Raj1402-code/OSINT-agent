"""
agent/planner.py
================================================================================
Investigation planning module for the OSINT Investigative Agent.

Takes a broad investigative question from the user and asks Gemini to break
it down into focused, independently-searchable sub-questions covering
different angles (background, stakeholders, recent developments, etc).
================================================================================
"""

from __future__ import annotations

import json
from typing import List

import google.generativeai as genai
from pydantic import BaseModel, Field

from agent.utils import get_logger

logger = get_logger(__name__)


class SubQuestion(BaseModel):
    question: str = Field(..., description="A specific, independently-searchable sub-question.")
    rationale: str = Field(default="", description="Why this sub-question matters.")


class InvestigationPlan(BaseModel):
    original_question: str = Field(...)
    sub_questions: List[SubQuestion] = Field(default_factory=list)


class InvestigationPlanner:
    """Uses Gemini to decompose a broad investigative question into focused sub-questions."""

    SYSTEM_PROMPT = """You are an expert OSINT (Open-Source Intelligence) investigation planner.

Given an investigative question from a user, break it down into a small set
of focused, independently-searchable sub-questions that, together, would let
a thorough investigator build a well-sourced, evidence-based answer.

GUIDELINES:
- Generate between 2 and {max_subquestions} sub-questions. Use fewer for
  simple/narrow questions, more for broad/complex ones.
- Each sub-question must be specific enough to type directly into a search
  engine and get useful, relevant results.
- Cover different angles: factual background, different stakeholder
  perspectives, recent developments, official statements, and likely
  points of controversy or disagreement.
- Do NOT generate leading/loaded sub-questions that presume an answer.
- Do NOT generate duplicate or near-duplicate sub-questions.
- Output ONLY valid JSON matching this schema, nothing else:

{{
  "sub_questions": [
    {{"question": "string", "rationale": "string - one sentence"}}
  ]
}}
"""

    def __init__(self, api_key: str, model: str, max_subquestions: int = 5) -> None:
        """
        Args:
            api_key: Gemini API key (genai.configure is called once at app startup;
                     kept here for interface clarity/future per-instance use).
            model: Gemini model name, e.g. "gemini-2.0-flash".
            max_subquestions: Upper bound on generated sub-questions.
        """
        self.model_name = model
        self.max_subquestions = max_subquestions

    def plan(self, original_question: str) -> InvestigationPlan:
        """Generate an investigation plan for a user question. Never raises."""
        if not original_question or not original_question.strip():
            raise ValueError("Cannot plan an investigation for an empty question.")

        system_prompt = self.SYSTEM_PROMPT.format(max_subquestions=self.max_subquestions)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
            )
            response = model.generate_content(
                f"Investigative question: {original_question}",
                generation_config={"response_mime_type": "application/json"},
            )
            raw_text = (response.text or "").strip()
            parsed = self._parse_json_response(raw_text)
            sub_questions = self._build_subquestions(parsed)

            if not sub_questions:
                logger.warning(
                    "Planner returned zero valid sub-questions; falling back to original question."
                )
                sub_questions = [
                    SubQuestion(
                        question=original_question,
                        rationale="Fallback: direct search of original question.",
                    )
                ]

            sub_questions = sub_questions[: self.max_subquestions]
            logger.info(f"Generated {len(sub_questions)} sub-question(s) for: '{original_question}'")
            return InvestigationPlan(original_question=original_question, sub_questions=sub_questions)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"Planning failed for question '{original_question}': {exc}. Falling back to direct search."
            )
            fallback = SubQuestion(
                question=original_question,
                rationale="Fallback: planner failed, searching original question directly.",
            )
            return InvestigationPlan(original_question=original_question, sub_questions=[fallback])

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict:
        """Defensively parse Gemini's JSON output, stripping markdown fences if present."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "sub_questions" in parsed:
                return parsed
            logger.warning(f"Unexpected JSON shape from planner model: {cleaned[:200]}")
            return {"sub_questions": []}
        except json.JSONDecodeError as exc:
            logger.warning(f"Failed to parse JSON from planner model: {exc}. Raw: {cleaned[:200]}")
            return {"sub_questions": []}

    @staticmethod
    def _build_subquestions(parsed: dict) -> List[SubQuestion]:
        sub_questions: List[SubQuestion] = []
        for item in parsed.get("sub_questions", []):
            question = (item.get("question") or "").strip()
            rationale = (item.get("rationale") or "").strip()
            if not question:
                continue
            try:
                sub_questions.append(SubQuestion(question=question, rationale=rationale))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Skipping malformed sub-question {item!r}: {exc}")
                continue
        return sub_questions
