"""
backend/orchestration/adapters/rag_adapter.py

Layer 3 — service recommendation, Gemini-backed.

Given the redacted evidence and the SVI result, Gemini suggests which
NHAA service track the case should be routed to and why.

TWO THINGS THIS MODULE DELIBERATELY DOES NOT LET THE MODEL DECIDE
-----------------------------------------------------------------
1. *Which routes exist.* Routes are an allow-list (`ROUTES`). Anything
   Gemini returns outside that list is rejected rather than passed
   through to the dashboard, so the model can't invent a service track
   the helpline doesn't actually operate.
2. *Whether a human has to confirm.* `requires_operator_confirmation`
   is computed from the risk tier in code, not taken from the model.
   High and Critical always require operator confirmation. The model
   may never talk the system out of a human check -- that flag is read
   directly by run.py's dashboard view and drives what the operator
   sees, so it is not a model opinion.

Contract (unchanged):
    run_rag(evidence_bundle, svi_result, rag_fn=None, timeout_seconds=...)
        -> (result|None, error|None)
"""

from __future__ import annotations

from typing import Callable, Optional

from backend.orchestration import errors
from backend.orchestration.adapters.gemini_client import call_gemini_json, evidence_to_payload
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.evidence_bundle import EvidenceBundle

RagFn = Callable[[EvidenceBundle, dict], dict]

ROUTES = [
    "counselling",
    "legal_aid",
    "medical",
    "police",
    "witness_protection",
]

# Tiers at which a human operator must confirm before anything proceeds.
_CONFIRMATION_TIERS = {"High", "Critical"}

_RAG_SYSTEM_TEXT = f"""
You are recommending which support services an NHAA (National Helpline
Against Atrocities, India) caller should be routed to. You receive a
PII-redacted transcript, explicit danger markers, tone cues, and an
already-computed risk assessment.

Choose one or more service tracks from exactly this list, and nothing
outside it: {", ".join(ROUTES)}

Return:
  recommended_routes: the tracks, most important first.
  rationale: 1-2 plain sentences an operator can read.
  urgency_note: one short line on timing, or an empty string.

Guidance:
- Base the recommendation on what the caller actually described.
- Immediate physical danger points toward police and, where the caller
  fears retaliation for reporting, witness_protection.
- Injury or assault described points toward medical.
- Nearly every caller benefits from counselling; include it unless
  clearly irrelevant.
- You are suggesting options to a trained human operator who makes the
  real decision. Never phrase this as a final or automatic action, and
  never tell the caller what to do directly.
"""

_RAG_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "recommended_routes": {
            "type": "array",
            "items": {"type": "string", "enum": ROUTES},
        },
        "rationale": {"type": "string"},
        "urgency_note": {"type": "string"},
    },
    "required": ["recommended_routes", "rationale"],
}


def _gemini_rag(evidence_bundle: EvidenceBundle, svi_result: dict) -> dict:
    payload = evidence_to_payload(evidence_bundle)
    payload["risk_assessment"] = {
        "risk_tier": svi_result.get("risk_tier"),
        "svi_score": svi_result.get("svi_score"),
        "rationale": svi_result.get("rationale"),
    }

    raw = call_gemini_json(
        system_text=_RAG_SYSTEM_TEXT,
        user_payload=payload,
        response_schema=_RAG_SCHEMA,
    )

    # Drop anything outside the allow-list rather than trusting the enum
    # was honoured -- structured output is a strong hint, not a guarantee.
    routes = [r for r in raw.get("recommended_routes", []) if r in ROUTES]
    if not routes:
        routes = ["counselling"]

    tier = svi_result.get("risk_tier")
    return {
        "recommended_routes": routes,
        "recommended_route": routes[0],  # kept for callers reading a single route
        "rationale": raw.get("rationale", ""),
        "urgency_note": raw.get("urgency_note", ""),
        # Computed here, never taken from the model. See module docstring.
        "requires_operator_confirmation": tier in _CONFIRMATION_TIERS,
        "basis": "gemini_recommended_route_allowlisted",
    }


def run_rag(
    evidence_bundle: EvidenceBundle,
    svi_result: dict,
    *,
    rag_fn: Optional[RagFn] = None,
    timeout_seconds: float = 10.0,
) -> tuple[Optional[dict], Optional[dict]]:
    fn = rag_fn or _gemini_rag
    try:
        result = run_with_timeout(lambda: fn(evidence_bundle, svi_result), seconds=timeout_seconds)
        return result, None
    except NodeTimeoutError as exc:
        return None, errors.stage_error("rag", str(exc), code=errors.TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return None, errors.stage_error("rag", f"RAG failed: {exc}")
