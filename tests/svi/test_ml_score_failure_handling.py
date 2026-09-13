"""
Layer 2 (SVI / Risk) — regression tests for a malfunctioning MLScorer's
return value.

`MLScorer.score()` (services/svi/ml/base.py) is documented to return a
float in [0, 100] or None, and to never let an exception escape. These
tests exercise what calculate_svi does when a *misbehaving*
implementation violates that contract anyway -- without itself
raising -- by returning NaN or a non-numeric value. Both must degrade
to `ml_score = None` (the same safe fallback as an absent/raising
scorer), never a silently-wrong numeric score and never an uncaught
exception out of calculate_svi.
"""

from __future__ import annotations

import math

from services.fusion import EvidenceBundle
from services.svi import RiskTier, calculate_svi
from services.svi.ml.base import MLScorer


class _ReturnsValueScorer(MLScorer):
    """Test double that returns whatever it's given, without raising --
    simulating a scorer implementation that violates its own interface
    contract instead of catching its internal failure."""

    def __init__(self, value):
        self._value = value

    def score(self, features):  # noqa: ANN001 - test double
        return self._value


def _empty_bundle(case_id: str) -> EvidenceBundle:
    return EvidenceBundle(case_id=case_id, source_input_ids=["IN-TEST"], markers=[], pii_redacted=True)


def test_nan_ml_score_degrades_to_none_not_100():
    """A NaN return must never silently become the maximum score."""
    bundle = _empty_bundle("CASE-BUG-NAN")

    result = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer(float("nan")))

    assert result.ml_score is None
    assert result.svi_score == 0.0  # falls back to rule_floor (0.0, no markers)
    assert result.risk_tier == RiskTier.LOW
    assert all(item.feature != "ml_score" for item in result.explanation)


def test_negative_nan_ml_score_also_degrades_to_none():
    bundle = _empty_bundle("CASE-BUG-NEG-NAN")

    result = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer(-float("nan")))

    assert result.ml_score is None
    assert math.isnan(float("nan"))  # sanity: nan is what we think it is


def test_non_numeric_ml_score_degrades_to_none_not_an_exception():
    """A scorer returning a non-float-convertible value must not raise
    out of calculate_svi -- it must degrade to ml_score=None like any
    other scorer failure."""
    bundle = _empty_bundle("CASE-BUG-NON-NUMERIC")

    result = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer("high"))

    assert result.ml_score is None
    assert result.svi_score == 0.0
    assert result.risk_tier == RiskTier.LOW


def test_non_numeric_object_ml_score_degrades_to_none():
    bundle = _empty_bundle("CASE-BUG-OBJECT")

    result = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer(object()))

    assert result.ml_score is None


def test_valid_numeric_ml_score_still_clamped_correctly():
    """Regression guard: the fix must not break the existing, already-
    correct clamping of ordinary out-of-range numeric scores."""
    bundle = _empty_bundle("CASE-BUG-STILL-CLAMPS")

    high = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer(150.0))
    low = calculate_svi(bundle, ml_scorer=_ReturnsValueScorer(-50.0))

    assert high.ml_score == 100.0
    assert low.ml_score == 0.0
