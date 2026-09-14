"""
Layer 4B (Orchestration) adapter for Layer 2 (SVI / Risk).

UPDATED: services/svi/ is now REAL (Member 3's feature/svi merge). This
adapter prefers `services.svi.svi_service.calculate_svi` and falls back
to `MockSVIProvider` only if that import ever fails again (e.g. running
against an older checkout, or in isolated tests that inject their own
provider).

Two contract facts from the real implementation that this adapter must
respect, verified by actually running it rather than assumed:

  1. `calculate_svi` requires an actual `services.fusion.EvidenceBundle`
     instance -- it raises `SviError(INVALID_INPUT)` for anything else,
     including a plain dict/mapping. An earlier version of this adapter
     converted the bundle to a mapping before calling the provider,
     which broke the moment the real implementation landed (11 tests
     failed with SVI_FAILED). Fixed here: the real EvidenceBundle object
     is passed through unconverted.
  2. `calculate_svi` returns a real `SVIResult` Pydantic model, not a
     dict -- normalized to a dict via `.model_dump(mode="json")` here so
     everything downstream (RAG/Support/dashboard) keeps working with
     plain dicts, unchanged.

`SviError` is now caught specifically (not just a bare `Exception`), so
a validation failure in Layer 2 is distinguishable in principle from an
unexpected crash -- both still degrade to the same safe `SVI_FAILED`
code today, but the distinction is available if the team ever wants
finer-grained codes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from backend.core.logging import log_event
from backend.orchestration.errors import stage_error
from services.fusion.evidence_bundle import EvidenceBundle

_VALID_TIERS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
_TIER_RANGES = ((0, 24, "LOW"), (25, 49, "MODERATE"), (50, 79, "HIGH"), (80, 100, "CRITICAL"))

# Kept only for MockSVIProvider's fallback behavior. Matches the marker
# `type` vocabulary in services/fusion/markers/marker_extractor.py, and
# now also matches Member 3's real services/svi/rules.py critical-marker
# table (self_harm, weapon, immediate_danger, threat, violence) -- this
# mock is retained only as a safety net if the real import ever fails.
_CRITICAL_MARKER_TYPES = {"self_harm", "immediate_danger", "weapon", "threat", "violence"}


def tier_for_score(score: float) -> str:
    for lo, hi, name in _TIER_RANGES:
        if lo <= score <= hi:
            return name
    raise ValueError("svi_score out of the valid 0-100 range")


class SVIProvider(Protocol):
    def calculate(self, evidence_bundle: EvidenceBundle) -> Any: ...


class MockSVIProvider:
    """
    Deterministic, contract-shaped FALLBACK for CONTRACTS.md 6.3, used
    only if `services.svi.svi_service.calculate_svi` cannot be imported.
    Not Member 3's real rule/ML layer. Takes the real EvidenceBundle
    object (attribute access), for the same input shape as the real
    provider, and returns a plain dict.
    """

    def calculate(self, evidence_bundle: EvidenceBundle) -> dict:
        markers = evidence_bundle.markers or []
        has_critical_marker = any(
            m.type in _CRITICAL_MARKER_TYPES and (m.confidence or 0) >= 0.6 for m in markers
        )
        ml_score = 0.0
        explanation: list[dict] = []
        for m in markers:
            impact = round((m.confidence or 0) * 20, 1)
            ml_score += impact
            explanation.append({"feature": m.type, "impact": impact})
        ml_score = max(0.0, min(round(ml_score, 1), 100.0))
        rule_floor = 80.0 if has_critical_marker else 0.0
        if has_critical_marker:
            explanation.append({"feature": "critical_safety_override", "impact": rule_floor})
        final_score = max(rule_floor, ml_score)

        return {
            "schema_version": "1.0.0",
            "case_id": evidence_bundle.case_id,
            "svi_score": final_score,
            "risk_tier": tier_for_score(final_score),
            "rule_floor": rule_floor,
            "ml_score": ml_score,
            "explanation": explanation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _load_real_provider() -> Optional[SVIProvider]:
    try:
        from services.svi.svi_service import calculate_svi  # type: ignore[import]
    except Exception:
        return None

    class _RealSVIAdapter:
        def calculate(self, evidence_bundle: EvidenceBundle):
            return calculate_svi(evidence_bundle)

    return _RealSVIAdapter()


def _normalize_result(raw: Any) -> dict:
    """Real calculate_svi returns a Pydantic SVIResult; MockSVIProvider
    and test doubles may return a plain dict already. Accept both."""
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(raw, dict):
        return raw
    raise ValueError("SVI provider returned neither a Pydantic model nor a dict")


def run_svi(
    evidence_bundle: Any,
    *,
    provider: Optional[SVIProvider] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Returns (svi_result, error). Exactly one is non-None. Never fabricates
    a score on failure -- a malformed/non-EvidenceBundle input or a
    provider exception becomes a structured SVI_FAILED error.
    """
    try:
        from services.svi.errors import SviError  # real, now that Layer 2 exists
    except Exception:
        SviError = None  # type: ignore[assignment]

    active_provider = provider or _load_real_provider() or MockSVIProvider()
    used_mock = isinstance(active_provider, MockSVIProvider) and provider is None

    try:
        if not isinstance(evidence_bundle, EvidenceBundle):
            raise ValueError("SVI requires a real EvidenceBundle instance, not a mapping or other type.")

        raw_result = active_provider.calculate(evidence_bundle)
        result = _normalize_result(raw_result)

        for field in ("schema_version", "case_id", "svi_score", "risk_tier", "rule_floor", "explanation", "created_at"):
            if field not in result:
                raise ValueError(f"SVIResult missing required field: {field}")
        if not (0 <= result["svi_score"] <= 100):
            raise ValueError("svi_score out of range")
        if result["risk_tier"] not in _VALID_TIERS:
            raise ValueError("invalid risk_tier")

        log_event(
            "svi_result_produced",
            case_id=evidence_bundle.case_id,
            risk_tier=result["risk_tier"],
            used_mock_provider=used_mock,
        )
        return result, None
    except Exception as exc:
        if SviError is not None and isinstance(exc, SviError):
            return None, stage_error("svi", "SVI calculation rejected malformed input.")
        return None, stage_error("svi", "SVI calculation failed.")
