"""
Layer 2 (SVI / Risk) — SVIResult schema (CONTRACTS.md 6.3).

This is the shared object Members 4, 5, and 6 depend on. It mirrors the
style already used for the shared schema in
services/fusion/evidence_bundle.py: a plain Pydantic model with field-level
constraints and `extra="forbid"`, no custom cross-field validators (the
project does not use `@field_validator`/`@model_validator` anywhere yet).

Schema-only stage: this module defines the *shape* of SVIResult and the
risk-tier boundaries from CONTRACTS.md. It does not compute a score --
that is the responsibility of the `calculate_svi` adapter (EvidenceBundle
-> SVIResult), implemented separately in services/svi/svi_service.py.

SAFETY NOTE: SVIResult is an assistive vulnerability/stress
PRIORITIZATION signal for human operators. It is NOT a medical diagnosis,
a psychiatric diagnosis, a definitive trauma diagnosis, or proof that a
person is being truthful or untruthful.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from services.svi.errors import INVALID_INPUT, SviError

SVI_RESULT_SCHEMA_VERSION = "1.0.0"


class RiskTier(str, Enum):
    """Allowed `risk_tier` values (CONTRACTS.md 6.3), by ascending score band."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Ascending-score-checked in risk_tier_for_score(); kept as an explicit,
# ordered table (rather than inline if/elif) so the boundaries are easy to
# audit against CONTRACTS.md: LOW 0-24, MODERATE 25-49, HIGH 50-79,
# CRITICAL 80-100.
_RISK_TIER_LOWER_BOUNDS: tuple[tuple[float, RiskTier], ...] = (
    (80.0, RiskTier.CRITICAL),
    (50.0, RiskTier.HIGH),
    (25.0, RiskTier.MODERATE),
    (0.0, RiskTier.LOW),
)


def risk_tier_for_score(score: float) -> RiskTier:
    """
    Map a 0-100 score to its CONTRACTS.md risk tier.

    Validates its own input (`0 <= score <= 100`) before mapping. This
    is a standalone defensive check on this function -- it does not
    change how calculate_svi computes svi_score, which already clamps
    final_score to [0, 100] before ever calling this helper.

    Pure otherwise: not wired into SVIResult's own schema validation
    (assigning a tier from a score is scoring-adjacent logic owned by
    calculate_svi, not a shape constraint on the SVIResult model), and
    independent of the dashboard/orchestration layers.
    """
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise SviError(INVALID_INPUT, f"risk_tier_for_score requires a numeric score, got {score!r}")
    if score != score:  # NaN check without importing math
        raise SviError(INVALID_INPUT, "risk_tier_for_score requires a real number, got NaN")
    if score < 0.0 or score > 100.0:
        raise SviError(INVALID_INPUT, f"risk_tier_for_score requires 0 <= score <= 100, got {score!r}")

    for lower_bound, tier in _RISK_TIER_LOWER_BOUNDS:
        if score >= lower_bound:
            return tier
    raise AssertionError("unreachable: score was validated to be within [0, 100]")  # pragma: no cover


class SVIExplanationItem(BaseModel):
    """
    One explainable contribution to `SVIResult.svi_score`.

    `feature` names the contributing signal (e.g. "marker:self_harm",
    "ml_score"); `impact` is that signal's contribution in the same 0-100
    units as `svi_score`. Generated from actual scoring features by
    calculate_svi, never invented after the fact.
    """

    feature: str
    impact: float

    model_config = ConfigDict(extra="forbid")


class SVIResult(BaseModel):
    """
    Layer 2's explainable risk output (CONTRACTS.md 6.3).

    Producer: Member 3 (SVI / Risk).
    Consumers: Members 4-6 (Service/RAG, Support, Dashboard).

    Downstream members must depend on this shape only -- never on
    services/svi internals.
    """

    schema_version: str = SVI_RESULT_SCHEMA_VERSION
    case_id: str
    svi_score: float = Field(ge=0.0, le=100.0)
    risk_tier: RiskTier
    rule_floor: float = Field(ge=0.0, le=100.0)
    ml_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    explanation: list[SVIExplanationItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="forbid")
