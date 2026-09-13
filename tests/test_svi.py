"""
Schema-only tests for Layer 2's SVIResult contract (CONTRACTS.md 6.3).

Scope: these tests validate the *shape* of SVIResult (required fields,
bounds, allowed risk_tier values, defaults) and the risk_tier_for_score
boundary helper. They do not test scoring/rule-floor/ML logic --
calculate_svi does not exist yet at this stage.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from services.svi import (
    SVI_RESULT_SCHEMA_VERSION,
    RiskTier,
    SVIExplanationItem,
    SVIResult,
    risk_tier_for_score,
)


def _valid_kwargs(**overrides):
    result = {
        "case_id": "CASE-01AB23CD",
        "svi_score": 60.0,
        "risk_tier": RiskTier.HIGH,
        "rule_floor": 50.0,
        "ml_score": 60.0,
        "explanation": [{"feature": "marker:threat", "impact": 50.0}],
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------
def test_svi_result_valid_construction_matches_contracts_example():
    result = SVIResult(
        case_id="CASE-01AB23CD",
        svi_score=87,
        risk_tier="CRITICAL",
        rule_floor=80,
        ml_score=87,
        explanation=[{"feature": "explicit_threat", "impact": 31}],
    )
    assert result.schema_version == SVI_RESULT_SCHEMA_VERSION
    assert result.risk_tier == RiskTier.CRITICAL
    assert result.explanation[0].feature == "explicit_threat"
    assert isinstance(result.created_at, datetime)


@pytest.mark.parametrize("tier", ["LOW", "MODERATE", "HIGH", "CRITICAL"])
def test_svi_result_accepts_plain_string_risk_tier(tier):
    """risk_tier should accept plain strings the same way Channel/Modality do."""
    result = SVIResult(**_valid_kwargs(risk_tier=tier))
    assert result.risk_tier == RiskTier(tier)


def test_svi_result_ml_score_defaults_to_none():
    kwargs = _valid_kwargs()
    kwargs.pop("ml_score")
    result = SVIResult(**kwargs)
    assert result.ml_score is None


def test_svi_result_explanation_defaults_to_empty_list():
    kwargs = _valid_kwargs()
    kwargs.pop("explanation")
    result = SVIResult(**kwargs)
    assert result.explanation == []


def test_svi_result_created_at_defaults_to_now_utc():
    before = datetime.now(timezone.utc)
    result = SVIResult(**_valid_kwargs())
    after = datetime.now(timezone.utc)
    assert before <= result.created_at <= after


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_score", [-1, 100.1, 101, -0.01])
def test_svi_result_rejects_out_of_range_svi_score(bad_score):
    with pytest.raises(ValidationError):
        SVIResult(**_valid_kwargs(svi_score=bad_score))


@pytest.mark.parametrize("bad_floor", [-1, 100.1, 200])
def test_svi_result_rejects_out_of_range_rule_floor(bad_floor):
    with pytest.raises(ValidationError):
        SVIResult(**_valid_kwargs(rule_floor=bad_floor))


@pytest.mark.parametrize("bad_ml", [-5, 150])
def test_svi_result_rejects_out_of_range_ml_score(bad_ml):
    with pytest.raises(ValidationError):
        SVIResult(**_valid_kwargs(ml_score=bad_ml))


def test_svi_result_rejects_invalid_risk_tier_string():
    with pytest.raises(ValidationError):
        SVIResult(**_valid_kwargs(risk_tier="SEVERE"))


def test_svi_result_rejects_missing_required_field():
    kwargs = _valid_kwargs()
    kwargs.pop("case_id")
    with pytest.raises(ValidationError):
        SVIResult(**kwargs)


def test_svi_result_rejects_unknown_extra_fields():
    with pytest.raises(ValidationError):
        SVIResult(**_valid_kwargs(unexpected_field="not allowed"))


def test_svi_explanation_item_requires_feature_and_impact():
    with pytest.raises(ValidationError):
        SVIExplanationItem(feature="marker:threat")  # missing impact
    item = SVIExplanationItem(feature="marker:threat", impact=42.5)
    assert item.feature == "marker:threat"
    assert item.impact == 42.5


def test_svi_explanation_item_rejects_unknown_extra_fields():
    with pytest.raises(ValidationError):
        SVIExplanationItem(feature="marker:threat", impact=10, note="extra")


# ---------------------------------------------------------------------------
# risk_tier_for_score boundaries (CONTRACTS.md 6.3: LOW 0-24, MODERATE
# 25-49, HIGH 50-79, CRITICAL 80-100)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "score,expected",
    [
        (0, RiskTier.LOW),
        (24, RiskTier.LOW),
        (24.99, RiskTier.LOW),
        (25, RiskTier.MODERATE),
        (49, RiskTier.MODERATE),
        (49.99, RiskTier.MODERATE),
        (50, RiskTier.HIGH),
        (79, RiskTier.HIGH),
        (79.99, RiskTier.HIGH),
        (80, RiskTier.CRITICAL),
        (100, RiskTier.CRITICAL),
    ],
)
def test_risk_tier_for_score_boundaries(score, expected):
    assert risk_tier_for_score(score) is expected
