"""
Layer 2 (SVI / Risk) — Member 3's module.

Public interface for other layers:

    from services.svi import calculate_svi
    svi_result = calculate_svi(evidence_bundle)

`calculate_svi` wires together the deterministic rule engine
(services/svi/rules.py) and the pluggable ML scorer
(services/svi/ml/*) per CONTRACTS.md section 6.3's intended
`final_score = max(rule_floor, ml_score)` rule.

Downstream members should depend on `calculate_svi` and `SVIResult`
only -- never on services.svi internals, per CONTRACTS.md integration
rule 4.
"""

from services.svi.errors import INVALID_INPUT, SviError
from services.svi.svi_result import (
    SVI_RESULT_SCHEMA_VERSION,
    RiskTier,
    SVIExplanationItem,
    SVIResult,
    risk_tier_for_score,
)
from services.svi.svi_service import calculate_svi

__all__ = [
    "calculate_svi",
    "SVIResult",
    "SVIExplanationItem",
    "RiskTier",
    "SVI_RESULT_SCHEMA_VERSION",
    "risk_tier_for_score",
    "SviError",
    "INVALID_INPUT",
]
