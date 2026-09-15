"""
backend/orchestration/adapters/support_adapter.py

Layer 3 — operator support, Gemini-backed.

This produces guidance for the HUMAN OPERATOR taking the call: what to
watch for, what to ask next, what to avoid saying. It does not produce
anything shown to the caller and it does not decide anything.

Two things are computed in code rather than left to the model:
  - `operator_alert` follows the risk tier directly. Whether the
    operator's screen lights up is not a model opinion.
  - The caller-facing boundary: the prompt forbids drafting text to read
    to the caller. Operators are trained; scripted LLM phrasing on a
    live atrocity/trauma call is not something this module should be
    quietly introducing.

nodes.py treats this as the non-critical branch on purpose -- a failure
here is recorded and the pipeline continues. Nothing in the service or
risk path may depend on this output.

Contract (unchanged):
    run_support(evidence_bundle, svi_result, support_fn=None, timeout_seconds=...)
        -> (result|None, error|None)
"""

from __future__ import annotations

from typing import Callable, Optional

from backend.orchestration import errors
from backend.orchestration.adapters.gemini_client import call_gemini_json, evidence_to_payload
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.evidence_bundle import EvidenceBundle

SupportFn = Callable[[EvidenceBundle, dict], dict]

_ALERT_TIERS = {"High", "Critical"}

_SUPPORT_SYSTEM_TEXT = """
You are assisting a trained human operator on a live call at the
National Helpline Against Atrocities (NHAA, India). You receive a
PII-redacted transcript of the call so far, explicit danger markers,
tone cues, and a risk assessment.

Return brief, practical notes FOR THE OPERATOR:
  operator_notes: 2-4 short bullets on what stands out in this call.
  suggested_questions: 2-4 short questions the operator could ask next
    to clarify safety, location, or immediate needs.
  cautions: things to avoid or be careful about with this caller
    (for example, a caller who may not be able to speak freely).

Hard rules:
- Do NOT write a script or any wording to read aloud to the caller. The
  operator is trained; you are giving them situational notes only.
- Do NOT diagnose the caller or name any mental health condition.
- Do NOT state or imply that any action has already been taken, or that
  help is on the way.
- Stay strictly within what the transcript supports. Do not speculate
  about the caller's identity, caste, religion, or circumstances.
- If the transcript is too short or unclear to say anything useful, say
  so plainly instead of inventing detail.
"""

_SUPPORT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "operator_notes": {"type": "array", "items": {"type": "string"}},
        "suggested_questions": {"type": "array", "items": {"type": "string"}},
        "cautions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["operator_notes"],
}


def _gemini_support(evidence_bundle: EvidenceBundle, svi_result: dict) -> dict:
    payload = evidence_to_payload(evidence_bundle)
    payload["risk_assessment"] = {
        "risk_tier": svi_result.get("risk_tier"),
        "svi_score": svi_result.get("svi_score"),
        "rationale": svi_result.get("rationale"),
    }

    raw = call_gemini_json(
        system_text=_SUPPORT_SYSTEM_TEXT,
        user_payload=payload,
        response_schema=_SUPPORT_SCHEMA,
    )

    tier = svi_result.get("risk_tier")
    return {
        "operator_notes": raw.get("operator_notes", []),
        "suggested_questions": raw.get("suggested_questions", []),
        "cautions": raw.get("cautions", []),
        # Computed here, not taken from the model.
        "operator_alert": tier in _ALERT_TIERS,
        "basis": "gemini_operator_support",
    }


def run_support(
    evidence_bundle: EvidenceBundle,
    svi_result: dict,
    *,
    support_fn: Optional[SupportFn] = None,
    timeout_seconds: float = 5.0,
) -> tuple[Optional[dict], Optional[dict]]:
    fn = support_fn or _gemini_support
    try:
        result = run_with_timeout(lambda: fn(evidence_bundle, svi_result), seconds=timeout_seconds)
        return result, None
    except NodeTimeoutError as exc:
        return None, errors.stage_error("support", str(exc), code=errors.TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return None, errors.stage_error("support", f"Support failed: {exc}")
