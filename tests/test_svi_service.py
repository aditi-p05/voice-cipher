"""
Focused unit tests for the wired-together SVI scoring engine
(services/svi/svi_service.py::calculate_svi): rule engine + ML scorer ->
final_score = max(rule_floor, ml_score) -> canonical SVIResult.
"""

from __future__ import annotations

import pytest

from services.fusion import EvidenceBundle, Marker
from services.svi import RiskTier, SviError, calculate_svi
from services.svi.ml.base import MLScorer


def _bundle(markers=None, **overrides) -> EvidenceBundle:
    kwargs = {
        "case_id": "CASE-SVI0001",
        "source_input_ids": ["IN-1"],
        "markers": markers or [],
        "pii_redacted": True,
    }
    kwargs.update(overrides)
    return EvidenceBundle(**kwargs)


class _FixedScorer(MLScorer):
    """Deterministic fake ML scorer returning a fixed value."""

    def __init__(self, value: float | None):
        self._value = value

    def score(self, features):
        return self._value


# ---------------------------------------------------------------------------
# Low score
# ---------------------------------------------------------------------------
def test_low_score_for_no_markers_and_no_ml_signal():
    result = calculate_svi(_bundle(markers=[]), ml_scorer=_FixedScorer(None))

    assert result.svi_score == 0.0
    assert result.rule_floor == 0.0
    assert result.ml_score is None
    assert result.risk_tier is RiskTier.LOW


# ---------------------------------------------------------------------------
# Moderate score
# ---------------------------------------------------------------------------
def test_moderate_score_from_supporting_markers_alone():
    bundle = _bundle(
        markers=[
            Marker(type="unsafe", value="not safe", confidence=1.0),  # rule impact 35.0
        ]
    )
    result = calculate_svi(bundle, ml_scorer=_FixedScorer(None))

    assert result.rule_floor == 35.0
    assert result.svi_score == 35.0
    assert result.risk_tier is RiskTier.MODERATE


# ---------------------------------------------------------------------------
# ML score higher than rule floor
# ---------------------------------------------------------------------------
def test_ml_score_higher_than_rule_floor_wins():
    bundle = _bundle(markers=[Marker(type="fear", value="i am scared", confidence=0.5)])  # rule impact 15.0
    result = calculate_svi(bundle, ml_scorer=_FixedScorer(70.0))

    assert result.rule_floor == 15.0
    assert result.ml_score == 70.0
    assert result.svi_score == 70.0  # max(15.0, 70.0)
    assert result.risk_tier is RiskTier.HIGH

    # ML contribution is explained; the rule floor did not govern the
    # final score, so no rule_floor_applied entry should appear.
    features = {item.feature for item in result.explanation}
    assert "ml_score" in features
    assert "rule_floor_applied" not in features
    assert "final_score" in features


# ---------------------------------------------------------------------------
# Rule floor higher than ML score
# ---------------------------------------------------------------------------
def test_rule_floor_higher_than_ml_score_wins():
    bundle = _bundle(markers=[Marker(type="unsafe", value="not safe", confidence=1.0)])  # rule impact 35.0
    result = calculate_svi(bundle, ml_scorer=_FixedScorer(10.0))

    assert result.rule_floor == 35.0
    assert result.ml_score == 10.0
    assert result.svi_score == 35.0  # max(35.0, 10.0)
    assert result.risk_tier is RiskTier.MODERATE

    features = {item.feature for item in result.explanation}
    assert "ml_score" in features
    assert "rule_floor_applied" in features  # the floor is why the score is 35.0, not 10.0


# ---------------------------------------------------------------------------
# Critical marker forcing the floor
# ---------------------------------------------------------------------------
def test_critical_marker_forces_the_floor_even_with_low_ml_score():
    bundle = _bundle(markers=[Marker(type="self_harm", value="kill myself", confidence=1.0)])
    result = calculate_svi(bundle, ml_scorer=_FixedScorer(10.0))

    assert result.rule_floor == 90.0
    assert result.ml_score == 10.0
    assert result.svi_score == 90.0  # NOT 10.0
    assert result.risk_tier is RiskTier.CRITICAL

    explanation_by_feature = {item.feature: item.impact for item in result.explanation}
    assert explanation_by_feature["CRITICAL_SELF_HARM"] == 90.0
    assert explanation_by_feature["ml_score"] == 10.0
    assert explanation_by_feature["rule_floor_applied"] == 90.0
    assert explanation_by_feature["final_score"] == 90.0


# ---------------------------------------------------------------------------
# Malformed / empty evidence
# ---------------------------------------------------------------------------
def test_empty_evidence_bundle_yields_a_valid_low_result_not_a_crash():
    bundle = _bundle(markers=[], transcript=None, sentiment=None, voice_features=None)
    result = calculate_svi(bundle, ml_scorer=_FixedScorer(None))

    assert result.svi_score == 0.0
    assert result.risk_tier is RiskTier.LOW
    assert result.explanation  # still includes at least the final_score entry


@pytest.mark.parametrize("malformed", [None, "not-a-bundle", {"case_id": "CASE-X"}, 42])
def test_malformed_input_raises_a_safe_typed_error_not_a_fake_result(malformed):
    with pytest.raises(SviError):
        calculate_svi(malformed)


def test_calculate_svi_is_deterministic_with_the_default_mock_scorer():
    bundle = _bundle(markers=[Marker(type="fear", value="i am scared", confidence=0.6)])

    first = calculate_svi(bundle)
    second = calculate_svi(bundle)

    assert first.svi_score == second.svi_score
    assert first.rule_floor == second.rule_floor
    assert first.ml_score == second.ml_score
    assert first.risk_tier == second.risk_tier
    assert [item.model_dump() for item in first.explanation] == [item.model_dump() for item in second.explanation]
