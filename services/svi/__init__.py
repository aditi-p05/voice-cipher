"""
Layer 2 (SVI / Risk) — Member 3's module.

Schema-only stage: this currently exposes the `SVIResult` output schema
required by CONTRACTS.md section 6.3 (and the `RiskTier` enum /
`risk_tier_for_score` helper it depends on). The `calculate_svi` adapter
(EvidenceBundle -> SVIResult) is implemented separately; importing this
package at this stage does not yet provide a scoring function.

Downstream members should depend on `SVIResult` (and, once implemented,
`calculate_svi`) only -- never on services.svi internals, per
CONTRACTS.md integration rule 4.
"""

from services.svi.svi_result import (
    SVI_RESULT_SCHEMA_VERSION,
    RiskTier,
    SVIExplanationItem,
    SVIResult,
    risk_tier_for_score,
)

__all__ = [
    "SVIResult",
    "SVIExplanationItem",
    "RiskTier",
    "SVI_RESULT_SCHEMA_VERSION",
    "risk_tier_for_score",
]
