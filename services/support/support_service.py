"""
Layer 4B (Support Engine) — public adapter.

    EvidenceBundle + SVIResult -> decide_support(...) -> SupportDecision

This is the ONLY function orchestration (or anyone else) should call.
Everything else in services/support is an implementation detail.

Risk gating (Developer 5 task brief + CONTRACTS.md risk tiers):
  LOW / MODERATE -> support may be enabled.
  HIGH            -> restricted; service escalation takes priority.
  CRITICAL        -> disabled as an automated *primary* intervention;
                     the critical service/escalation pathway takes
                     priority and support must never delay it.

This function must never raise for merely-missing optional evidence
(no sentiment, no markers). It raises SupportError only for genuinely
malformed/missing required input, and callers (orchestration) must
catch that without letting a support failure affect the service path.
"""

from __future__ import annotations

from typing import Any, Mapping

from services.support.errors import INVALID_INPUT, SupportError
from services.support.models import NON_CLINICAL_DISCLAIMER, SupportDecision
from services.support.presets import select_preset

_REQUIRED_EVIDENCE_FIELDS = ("schema_version", "case_id", "markers", "pii_redacted", "created_at")
_REQUIRED_SVI_FIELDS = ("schema_version", "case_id", "svi_score", "risk_tier", "created_at")
_VALID_TIERS = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    raise SupportError(INVALID_INPUT, f"{label} must be a mapping or Pydantic model.")


def decide_support(evidence_bundle: Any, svi_result: Any) -> dict:
    """Return a contract-shaped, risk-gated support decision as a dict."""
    evidence = _as_mapping(evidence_bundle, "EvidenceBundle")
    svi = _as_mapping(svi_result, "SVIResult")

    for field in _REQUIRED_EVIDENCE_FIELDS:
        if field not in evidence:
            raise SupportError(INVALID_INPUT, f"EvidenceBundle missing required field: {field}")
    for field in _REQUIRED_SVI_FIELDS:
        if field not in svi:
            raise SupportError(INVALID_INPUT, f"SVIResult missing required field: {field}")
    if evidence["case_id"] != svi["case_id"]:
        raise SupportError(INVALID_INPUT, "EvidenceBundle and SVIResult case_id values must match.")

    risk_tier = svi["risk_tier"]
    if risk_tier not in _VALID_TIERS:
        raise SupportError(INVALID_INPUT, "SVIResult risk_tier is invalid.")

    case_id = evidence["case_id"]
    sentiment = evidence.get("sentiment") or {}
    valence = sentiment.get("valence") if isinstance(sentiment, Mapping) else None
    arousal = sentiment.get("arousal") if isinstance(sentiment, Mapping) else None

    if risk_tier == "CRITICAL":
        decision = SupportDecision(
            case_id=case_id,
            risk_tier=risk_tier,
            mode="disabled",
            support_enabled=False,
            preset=None,
            reason=(
                "Critical safety/service pathway takes priority; automated "
                "support is disabled as a primary intervention at this tier."
            ),
        )
    elif risk_tier == "HIGH":
        decision = SupportDecision(
            case_id=case_id,
            risk_tier=risk_tier,
            mode="restricted",
            support_enabled=False,
            preset=None,
            reason="Priority service escalation takes precedence at HIGH risk.",
        )
    else:
        preset = select_preset(valence, arousal)
        decision = SupportDecision(
            case_id=case_id,
            risk_tier=risk_tier,
            mode="enabled",
            support_enabled=True,
            preset=preset,
            reason=f"Support permitted at {risk_tier} risk based on available sentiment signal.",
            disclaimer=NON_CLINICAL_DISCLAIMER,
        )

    return decision.model_dump(mode="json")
