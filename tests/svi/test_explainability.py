"""
Layer 2 (SVI / Risk) — Step 6 focused tests: explainability.

These tests only cover the *explanation* output of `calculate_svi`
(services/svi/svi_service.py) -- not rule scoring, ML scoring, or risk
tier boundaries, which already have their own focused test files from
Steps 2-5. Synthetic EvidenceBundle fixtures only; no real evidence.
"""

from __future__ import annotations

from services.fusion import EvidenceBundle, Marker, Sentiment
from services.svi import SVIExplanationItem, calculate_svi
from services.svi.ml.base import MLScorer
from services.svi.ml.null_scorer import NullMLScorer


class _FixedScorer(MLScorer):
    """Deterministic test double: always returns a fixed score."""

    def __init__(self, value: float | None):
        self._value = value

    def score(self, features):  # noqa: ANN001 - test double
        return self._value


def _bundle(case_id: str = "CASE-EXPLAIN", markers=None, sentiment=None) -> EvidenceBundle:
    return EvidenceBundle(
        case_id=case_id,
        source_input_ids=["IN-TEST"],
        markers=markers or [],
        sentiment=sentiment,
        pii_redacted=True,
    )


def test_explanation_items_are_structured_data_not_free_text():
    """Every entry must be a proper SVIExplanationItem (feature/impact/
    description), never a freeform LLM-style paragraph."""
    bundle = _bundle(markers=[Marker(type="fear", value="scared", confidence=0.9)])

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    assert len(result.explanation) > 0
    for item in result.explanation:
        assert isinstance(item, SVIExplanationItem)
        assert isinstance(item.feature, str) and item.feature
        assert isinstance(item.impact, float)


def test_triggered_rule_carries_its_description_from_rules_py():
    """Rule descriptions must come straight from RuleHit, not be
    reinvented in svi_service.py."""
    bundle = _bundle(markers=[Marker(type="self_harm", value="explicit", confidence=1.0)])

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    rule_items = [item for item in result.explanation if item.feature == "CRITICAL_SELF_HARM"]
    assert len(rule_items) == 1
    assert rule_items[0].description == "Explicit self-harm marker present"


def test_ml_score_entry_present_with_description_when_ml_score_available():
    bundle = _bundle(markers=[])

    result = calculate_svi(bundle, ml_scorer=_FixedScorer(42.0))

    ml_items = [item for item in result.explanation if item.feature == "ml_score"]
    assert len(ml_items) == 1
    assert ml_items[0].impact == 42.0
    assert ml_items[0].description is not None


def test_no_ml_score_entry_when_scorer_unavailable():
    bundle = _bundle(markers=[Marker(type="fear", value="scared", confidence=0.5)])

    result = calculate_svi(bundle, ml_scorer=_FixedScorer(None))

    assert result.ml_score is None
    assert all(item.feature != "ml_score" for item in result.explanation)


def test_rule_floor_applied_entry_present_when_floor_governs():
    """Critical marker with no ML score -> rule floor governs the final
    score, so a rule_floor_applied entry must explain why."""
    bundle = _bundle(markers=[Marker(type="weapon", value="explicit", confidence=1.0)])

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    floor_items = [item for item in result.explanation if item.feature == "rule_floor_applied"]
    assert len(floor_items) == 1
    assert floor_items[0].impact == result.rule_floor
    assert floor_items[0].description is not None


def test_rule_floor_applied_entry_absent_when_ml_score_governs():
    """ML score above the rule floor -> nothing to explain as a floor
    override, so the rule_floor_applied entry must not appear."""
    bundle = _bundle(markers=[Marker(type="fear", value="scared", confidence=0.5)])

    result = calculate_svi(bundle, ml_scorer=_FixedScorer(90.0))

    assert result.svi_score == 90.0
    assert all(item.feature != "rule_floor_applied" for item in result.explanation)


def test_final_score_entry_always_present_and_last_before_risk_tier():
    bundle = _bundle(markers=[])

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    features = [item.feature for item in result.explanation]
    assert "final_score" in features
    assert features[-1] == "risk_tier"
    assert features[-2] == "final_score"

    final_item = next(item for item in result.explanation if item.feature == "final_score")
    assert final_item.impact == result.svi_score


def test_risk_tier_entry_present_and_matches_result_tier():
    bundle = _bundle(markers=[Marker(type="self_harm", value="explicit", confidence=1.0)])

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    tier_items = [item for item in result.explanation if item.feature == "risk_tier"]
    assert len(tier_items) == 1
    assert result.risk_tier.value in tier_items[0].description


def test_empty_evidence_still_yields_structured_final_and_tier_entries():
    """Missing/empty optional evidence must never raise -- explanation
    should still be well-formed, structured, low-risk data."""
    bundle = _bundle(markers=[], sentiment=None)

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    assert result.svi_score == 0.0
    features = [item.feature for item in result.explanation]
    assert features == ["final_score", "risk_tier"]
