"""
backend/orchestration/adapters/gemini_client.py

Shared Gemini call helper for the SVI / RAG / Support adapters.

Exists so the three adapters don't each re-implement client construction,
structured-output parsing, and the "SDK not installed / key not set"
error paths. The fusion adapter has its own call path because it sends
audio parts rather than a plain JSON-in/JSON-out text prompt.

Everything here raises on failure -- each adapter is responsible for
catching and converting to its own structured stage error, so the
"a stage failing must never bubble a raw exception" rule in nodes.py
stays intact.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.orchestration import config


def call_gemini_json(
    *,
    system_text: str,
    user_payload: dict[str, Any],
    response_schema: dict[str, Any],
) -> dict[str, Any]:
    """Send one JSON payload to Gemini and get structured JSON back."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env "
            "(backend/orchestration/config.py already reads it)."
        )

    from google import genai  # lazy import so a missing SDK only breaks Gemini stages

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    interaction = client.interactions.create(
        model=config.GEMINI_MODEL,
        input=[
            {"type": "text", "text": system_text},
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, default=str)},
        ],
        response_format=response_schema,
    )
    return parse_json_output(interaction.output_text)


def parse_json_output(output_text: str) -> dict[str, Any]:
    cleaned = (output_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def evidence_to_payload(evidence_bundle: Any) -> dict[str, Any]:
    """
    Flatten an EvidenceBundle into the minimal, already-PII-redacted shape
    we send to Gemini. Deliberately NOT `model_dump()` of the whole bundle:
    we only send what a scoring/recommendation decision actually needs, so
    no extra internal identifiers leave the system.
    """
    transcript = evidence_bundle.transcript
    return {
        "transcript": transcript.text if transcript else "",
        "language": transcript.language if transcript else None,
        "transcript_confidence": transcript.confidence if transcript else None,
        "pii_redacted": evidence_bundle.pii_redacted,
        "explicit_markers": [
            {
                "type": m.marker_type,
                "severity": m.severity,
                "matched_text": m.matched_text,
            }
            for m in evidence_bundle.markers
        ],
        "tone": (
            {
                "valence": evidence_bundle.sentiment.valence,
                "arousal": evidence_bundle.sentiment.arousal,
            }
            if evidence_bundle.sentiment
            else None
        ),
    }
