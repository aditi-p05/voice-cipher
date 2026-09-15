"""
backend/orchestration/adapters/svi_adapter.py

Layer 2 — Stress Vulnerability Index scoring, Gemini-backed.

Gemini reads the (already PII-redacted) transcript, the explicit
rule-based markers, and the tone cues, and returns an SVI score 0-100
with a risk tier and a short human-readable rationale the operator can
actually see.

WHY THERE IS A DETERMINISTIC FLOOR ON TOP OF THE MODEL
------------------------------------------------------
The model is the scorer, not the last word. `_apply_safety_floor`
raises the tier -- never lowers it -- when explicit markers were found:

    any "critical"-severity marker  -> tier at least Critical
    any "high"-severity marker      -> tier at least High

This exists because an LLM returning a soft tier for a transcript that
literally contains an explicit same-day death threat is a single-call
failure mode a helpline cannot absorb, and it is exactly the dilution
your own evidence_merge.py already refuses to allow across chunks. The
floor only ever moves risk UP, so a model that is correctly alarmed is
never overridden; a model that under-reacts on explicit evidence is.
When the floor fires it is recorded in the result (`safety_floor_applied`,
`model_risk_tier`) so it's visible rather than silent.

If Member 3 later ships a real quantitative SVI implementation, pass it
in as `Providers(svi_provider=their_fn)` -- this module is the default,
not a hard dependency.

Contract (unchanged, nodes.py depends on it):
    run_svi(evidence_bundle, provider=None) -> (result|None, error|None)
    result always contains "risk_tier".
"""

from __future__ import annotations

from typing import Callable, Optional

from backend.orchestration import config, errors
from backend.orchestration.adapters.gemini_client import call_gemini_json, evidence_to_payload
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.evidence_bundle import EvidenceBundle

SVIProvider = Callable[[EvidenceBundle], dict]

TIER_ORDER = ["Low", "Moderate", "High", "Critical"]

_SVI_SYSTEM_TEXT = """
You are scoring a caller's distress and vulnerability for the National
Helpline Against Atrocities (NHAA, India). You will receive a JSON object
containing a PII-redacted transcript of what the caller said, any explicit
danger markers a separate rule-based pass already found, and coarse tone
cues from the audio.

Return a Stress Vulnerability Index:
  svi_score: 0-100, where 100 is maximum distress and vulnerability.
  risk_tier: one of Low, Moderate, High, Critical.
  rationale: 1-2 plain sentences an operator can read, referring only to
    what is actually in the transcript or markers.
  key_factors: short list of the specific things that drove the score.

Scoring guidance:
- Explicit statements of danger, threats, or violence in the transcript
  are strong evidence and should dominate the score.
- Tone cues (fear, anger, distress) are supporting evidence only. A
  fearful tone alone, with nothing concerning said, is not High risk.
- A calm tone does NOT reduce the weight of an explicit threat. People
  disclose danger flatly.
- Do not speculate beyond what was said. Do not infer identity,
  caste, religion, or any attribute that is not stated.
- You are one input to a human operator's decision, not the decision.
  Never state or imply that a final action has been taken.
"""

_SVI_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "svi_score": {"type": "number"},
        "risk_tier": {"type": "string", "enum": TIER_ORDER},
        "rationale": {"type": "string"},
        "key_factors": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["svi_score", "risk_tier", "rationale"],
}


def _gemini_svi(bundle: EvidenceBundle) -> dict:
    raw = call_gemini_json(
        system_text=_SVI_SYSTEM_TEXT,
        user_payload=evidence_to_payload(bundle),
        response_schema=_SVI_SCHEMA,
    )

    model_tier = raw.get("risk_tier")
    if model_tier not in TIER_ORDER:
        raise ValueError(f"Gemini returned an unrecognised risk_tier: {model_tier!r}")

    score = float(raw.get("svi_score", 0.0) or 0.0)
    final_tier, floor_applied = _apply_safety_floor(model_tier, bundle)

    result = {
        "svi_score": round(max(0.0, min(100.0, score)), 1),
        "risk_tier": final_tier,
        "rationale": raw.get("rationale", ""),
        "key_factors": raw.get("key_factors", []),
        "basis": "gemini_scored",
    }
    if floor_applied:
        # Make the override visible rather than silently rewriting the model.
        result["safety_floor_applied"] = True
        result["model_risk_tier"] = model_tier
        result["basis"] = "gemini_scored_with_safety_floor"
    return result


def _apply_safety_floor(model_tier: str, bundle: EvidenceBundle) -> tuple[str, bool]:
    """Raise the tier to match explicit marker severity. Never lowers it."""
    severities = {m.severity for m in bundle.markers}
    if "critical" in severities:
        floor = "Critical"
    elif "high" in severities:
        floor = "High"
    else:
        return model_tier, False

    if TIER_ORDER.index(model_tier) >= TIER_ORDER.index(floor):
        return model_tier, False
    return floor, True


def run_svi(
    evidence_bundle: EvidenceBundle,
    *,
    provider: Optional[SVIProvider] = None,
    timeout_seconds: Optional[float] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    fn = provider or _gemini_svi
    budget = timeout_seconds if timeout_seconds is not None else config.SVI_TIMEOUT_SECONDS
    try:
        return run_with_timeout(lambda: fn(evidence_bundle), seconds=budget), None
    except NodeTimeoutError as exc:
        return None, errors.stage_error("svi", str(exc), code=errors.TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return None, errors.stage_error("svi", f"SVI failed: {exc}")
