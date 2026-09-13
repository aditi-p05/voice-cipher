"""
Focused unit tests for Layer 2's risk-tier mapping
(services/svi/svi_result.py::risk_tier_for_score).

Scope: tier mapping and score validation only. Does not touch, and
does not test, the SVI scoring algorithm (rule engine / ML scorer /
calculate_svi) -- see tests/test_svi_rules.py, tests/test_svi_ml.py,
and tests/test_svi_service.py for that. Independent of the dashboard
and orchestration layers, which this module does not import.
"""

from __future__ import annotations

import pytest

from services.svi import RiskTier, SviError, risk_tier_for_score

# ---------------------------------------------------------------------------
# Required boundary values: 0, 24, 25, 49, 50, 79, 80, 100
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected_tier",
    [
        (0, RiskTier.LOW),
        (24, RiskTier.LOW),
        (25, RiskTier.MODERATE),
        (49, RiskTier.MODERATE),
        (50, RiskTier.HIGH),
        (79, RiskTier.HIGH),
        (80, RiskTier.CRITICAL),
        (100, RiskTier.CRITICAL),
    ],
)
def test_risk_tier_boundary_values(score, expected_tier):
    assert risk_tier_for_score(score) is expected_tier


# ---------------------------------------------------------------------------
# Critical must be represented explicitly
# ---------------------------------------------------------------------------
def test_critical_tier_is_an_explicit_named_value():
    assert RiskTier.CRITICAL.value == "CRITICAL"
    assert risk_tier_for_score(80) is RiskTier.CRITICAL
    assert risk_tier_for_score(100) is RiskTier.CRITICAL


# ---------------------------------------------------------------------------
# Score must be validated
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_score", [-1, -0.01, 100.01, 101, 1000])
def test_out_of_range_score_is_rejected(bad_score):
    with pytest.raises(SviError):
        risk_tier_for_score(bad_score)


@pytest.mark.parametrize("bad_score", ["80", None, [80], {}, object()])
def test_non_numeric_score_is_rejected(bad_score):
    with pytest.raises(SviError):
        risk_tier_for_score(bad_score)


def test_boolean_score_is_rejected():
    # bool is a subclass of int in Python; must not be silently accepted
    # as a valid score (e.g. True == 1, which would otherwise pass).
    with pytest.raises(SviError):
        risk_tier_for_score(True)


def test_nan_score_is_rejected():
    with pytest.raises(SviError):
        risk_tier_for_score(float("nan"))


def test_valid_score_still_works_after_validation_is_added():
    assert risk_tier_for_score(0.0) is RiskTier.LOW
    assert risk_tier_for_score(100.0) is RiskTier.CRITICAL
