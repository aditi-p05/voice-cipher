"""Recommendation generators. Provider output is constrained to retrieved SOP text."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, Sequence

from .models import RetrievalMatch


class RecommendationGenerator(Protocol):
    def generate(self, context: Sequence[RetrievalMatch], query: str) -> list[dict[str, Any]]: ...


class DeterministicRecommendationGenerator:
    """Demo fallback: return only explicit SOP bullet actions, never inferred procedure."""
    def generate(self, context: Sequence[RetrievalMatch], query: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for match in context:
            actions = re.findall(r"^\s*[-*]\s+(?:Action:\s*)?(.+?)\s*$", match.chunk.text, re.M | re.I)
            if not actions:
                continue
            action = actions[0]
            results.append({
                "action": action,
                "reason": "Retrieved SOP section is relevant to the redacted evidence and risk tier.",
                "source": match.chunk.citation,
                "confidence": round(min(1.0, max(0.0, match.score)), 2),
            })
        return results


class OpenAIRecommendationGenerator:
    """Optional provider adapter. It is selected only with an environment API key."""
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

    def generate(self, context: Sequence[RetrievalMatch], query: str) -> list[dict[str, Any]]:
        from openai import OpenAI  # optional dependency, intentionally lazy
        passages = "\n\n".join(f"[{m.chunk.citation}]\n{m.chunk.text}" for m in context)
        prompt = ("Only recommend actions supported by the retrieved SOP context. If the SOP context "
                  "does not support a recommendation, return []. Do not invent legal, police, medical, "
                  "emergency, or government procedures. Return JSON list with action, reason, source, confidence.\n"
                  f"Query: {query}\nSOP context:\n{passages}")
        response = OpenAI(api_key=self.api_key).chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
        payload = json.loads(response.choices[0].message.content or "{}")
        items = payload.get("recommendations", payload if isinstance(payload, list) else [])
        citations = {match.chunk.citation: match for match in context}
        # Reject provider output that cannot be traced to its source passage.
        return [item for item in items if isinstance(item, dict) and item.get("source") in citations
                and item.get("action", "").lower() in citations[item["source"]].chunk.text.lower()]
