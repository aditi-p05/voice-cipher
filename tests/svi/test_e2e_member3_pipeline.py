"""
Layer 2 (SVI / Risk) — Step 7: end-to-end integration test for Member 3's
own pipeline ONLY.

    EvidenceBundle -> calculate_svi -> SVIResult -> risk_tier -> explanation

Per CONTRACTS.md 6.2/6.3, Member 3 depends only on the `EvidenceBundle`
shape, never on Member 2's internals (services.fusion.fusion_service /
build_evidence_bundle) or on Member 1's InputEnvelope. So this file
builds EvidenceBundle instances directly -- synthetic/mock data, exactly
as the Step 7 brief and CONTRACTS.md's own "Member 3 should work
independently using synthetic EvidenceBundle data" checkpoint require.

This file does not import or exercise services.rag (Member 4),
services.support (Member 5), or any dashboard/API code (Member 6) --
those consume SVIResult, but Member 3's own integration test does not
need a live producer of that data.

Scope note (per CONTRACTS.md 6.3 / Step 5): risk tier bands are
LOW 0-24, MODERATE 25-49, HIGH 50-79, CRITICAL 80-100.
"""

from __future__ import annotations

import math

import pytest

from services.fusion import EvidenceBundle, Marker, Sentiment, VoiceFeatures
from services.svi import (
    RiskTier,
    SVIExplanationItem,
    SVIResult,
    SviError,
    calculate_svi,
    risk_tier_for_score,
)
from services.svi.ml.base import MLScorer
from services.svi.ml.null_scorer import NullMLScorer


class _FixedScorer(MLScorer):
    """Deterministic test double: always returns a fixed (possibly
    out-of-range) score, to exercise calculate_svi's own clamping."""

    def __init__(self, value: float | None):
        self._value = value

    def score(self, features):  # noqa: ANN001 - test double
        return self._value


def _bundle(
    case_id: str = "CASE-E2E",
    markers=None,
    sentiment=None,
    voice_features=None,
    transcript=None,
    source_input_ids=None,
) -> EvidenceBundle:
    return EvidenceBundle(
        case_id=case_id,
        source_input_ids=source_input_ids or ["IN-SYNTHETIC"],
        transcript=transcript,
        markers=markers or [],
        sentiment=sentiment,
        voice_features=voice_features,
        pii_redacted=True,
    )


def _assert_well_formed_pipeline_output(result: SVIResult, bundle: EvidenceBundle) -> None:
    """Shape assertions shared by every case: EvidenceBundle -> SVIResult
    -> risk_tier -> explanation, all structured, all contract-shaped."""
    assert isinstance(result, SVIResult)
    assert result.case_id == bundle.case_id
    assert 0.0 <= result.svi_score <= 100.0
    assert 0.0 <= result.rule_floor <= 100.0
    assert result.ml_score is None or 0.0 <= result.ml_score <= 100.0
    assert isinstance(result.risk_tier, RiskTier)
    # risk_tier must actually correspond to the returned svi_score.
    assert result.risk_tier == risk_tier_for_score(result.svi_score)
    assert isinstance(result.explanation, list)
    assert len(result.explanation) > 0
    for item in result.explanation:
        assert isinstance(item, SVIExplanationItem)
    assert result.explanation[-1].feature == "risk_tier"
    assert result.explanation[-2].feature == "final_score"


# ---------------------------------------------------------------------
# 1. Low-risk case
# ---------------------------------------------------------------------
def test_low_risk_case_end_to_end():
    bundle = _bundle(
        case_id="CASE-E2E-LOW",
        markers=[Marker(type="fear", value="mild unease", confidence=0.3)],
        sentiment=Sentiment(valence=0.1, arousal=0.2),
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.risk_tier == RiskTier.LOW
    assert result.svi_score < 25.0
    assert result.ml_score is None


# ---------------------------------------------------------------------
# 2. Moderate-risk case
# ---------------------------------------------------------------------
def test_moderate_risk_case_end_to_end():
    bundle = _bundle(
        case_id="CASE-E2E-MODERATE",
        markers=[Marker(type="coercion", value="pressured", confidence=0.6)],
        sentiment=Sentiment(valence=-0.3, arousal=0.4),
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.risk_tier == RiskTier.MODERATE
    assert 25.0 <= result.svi_score <= 49.0


# ---------------------------------------------------------------------
# 3. High-risk case
# ---------------------------------------------------------------------
def test_high_risk_case_end_to_end():
    # Three co-occurring supporting markers at full confidence:
    # primary impact 45.0 (coercion) + capped co-occurrence bonus 6.0
    # (3 distinct types) = 51.0 -> HIGH (50-79).
    bundle = _bundle(
        case_id="CASE-E2E-HIGH",
        markers=[
            Marker(type="coercion", value="pressured", confidence=1.0),
            Marker(type="retaliation", value="threatened consequences", confidence=1.0),
            Marker(type="unsafe", value="unsafe environment", confidence=1.0),
        ],
        sentiment=Sentiment(valence=-0.6, arousal=0.7),
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.risk_tier == RiskTier.HIGH
    assert 50.0 <= result.svi_score <= 79.0
    # Multiple distinct marker types should be visible in the explanation.
    assert any(item.feature == "MULTIPLE_CONCERNING_MARKERS" for item in result.explanation)


# ---------------------------------------------------------------------
# 4. Critical case
# ---------------------------------------------------------------------
def test_critical_case_end_to_end():
    bundle = _bundle(
        case_id="CASE-E2E-CRITICAL",
        markers=[Marker(type="self_harm", value="explicit statement", confidence=1.0)],
        sentiment=Sentiment(valence=-0.9, arousal=0.95),
        voice_features=VoiceFeatures(jitter=0.05, shimmer=0.08),
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.risk_tier == RiskTier.CRITICAL
    assert result.svi_score >= 80.0
    assert any(item.feature == "CRITICAL_SELF_HARM" for item in result.explanation)
    assert any(item.feature == "rule_floor_applied" for item in result.explanation)


# ---------------------------------------------------------------------
# 5. Critical marker overriding a lower ML score
# ---------------------------------------------------------------------
def test_critical_marker_overrides_lower_ml_score():
    """Safety rule (CONTRACTS.md 6.3): final_score = max(rule_floor,
    ml_score). A critical rule floor must never be pulled down by a
    lower ML score."""
    bundle = _bundle(
        case_id="CASE-E2E-FLOOR-OVERRIDE",
        markers=[Marker(type="weapon", value="explicit mention", confidence=1.0)],
    )
    low_ml_scorer = _FixedScorer(20.0)

    result = calculate_svi(bundle, ml_scorer=low_ml_scorer)

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.rule_floor == 88.0
    assert result.ml_score == 20.0
    assert result.svi_score == 88.0  # rule floor wins, not the ML score
    assert result.risk_tier == RiskTier.CRITICAL
    assert any(item.feature == "rule_floor_applied" for item in result.explanation)
    # The lower ML score must still be visible/explainable, just non-governing.
    assert any(item.feature == "ml_score" and item.impact == 20.0 for item in result.explanation)


# ---------------------------------------------------------------------
# 6. Empty / partial evidence
# ---------------------------------------------------------------------
def test_completely_empty_evidence_does_not_raise():
    bundle = _bundle(case_id="CASE-E2E-EMPTY", markers=[], sentiment=None, voice_features=None, transcript=None)

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.svi_score == 0.0
    assert result.rule_floor == 0.0
    assert result.ml_score is None
    assert result.risk_tier == RiskTier.LOW
    assert [item.feature for item in result.explanation] == ["final_score", "risk_tier"]


def test_partial_evidence_sentiment_only_no_markers():
    """Voice/text evidence present, no markers extracted yet -- must
    still yield a valid, low-scoring SVIResult, not an error."""
    bundle = _bundle(
        case_id="CASE-E2E-PARTIAL",
        markers=[],
        sentiment=Sentiment(valence=-0.4, arousal=0.5),
        voice_features=None,
        transcript=None,
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.rule_floor == 0.0
    assert result.risk_tier == RiskTier.LOW


def test_partial_evidence_audio_only_no_transcript():
    """CONTRACTS.md 6.2: transcript: null is legitimate for
    audio/voice-features-only evidence; downstream must accept it."""
    bundle = _bundle(
        case_id="CASE-E2E-AUDIO-ONLY",
        markers=[],
        transcript=None,
        voice_features=VoiceFeatures(pitch_hz_stdev=12.0, jitter=0.02),
    )

    result = calculate_svi(bundle, ml_scorer=NullMLScorer())

    _assert_well_formed_pipeline_output(result, bundle)
    assert result.risk_tier == RiskTier.LOW


# ---------------------------------------------------------------------
# 7. Invalid score boundary
# ---------------------------------------------------------------------
def test_risk_tier_for_score_rejects_out_of_range_scores():
    with pytest.raises(SviError) as exc_info_high:
        risk_tier_for_score(100.1)
    assert exc_info_high.value.code == "INVALID_INPUT"

    with pytest.raises(SviError) as exc_info_low:
        risk_tier_for_score(-0.1)
    assert exc_info_low.value.code == "INVALID_INPUT"


def test_risk_tier_for_score_rejects_nan():
    with pytest.raises(SviError) as exc_info:
        risk_tier_for_score(float("nan"))
    assert exc_info.value.code == "INVALID_INPUT"


def test_calculate_svi_clamps_out_of_range_ml_scores_before_boundary_check():
    """A misbehaving MLScorer returning an out-of-[0,100] value must be
    clamped by calculate_svi itself before it ever reaches
    risk_tier_for_score -- the pipeline must never propagate an invalid
    boundary value downstream."""
    bundle = _bundle(case_id="CASE-E2E-CLAMP-HIGH", markers=[])
    result_high = calculate_svi(bundle, ml_scorer=_FixedScorer(150.0))
    _assert_well_formed_pipeline_output(result_high, bundle)
    assert result_high.ml_score == 100.0
    assert result_high.svi_score == 100.0
    assert result_high.risk_tier == RiskTier.CRITICAL

    result_low = calculate_svi(bundle, ml_scorer=_FixedScorer(-50.0))
    _assert_well_formed_pipeline_output(result_low, bundle)
    assert result_low.ml_score == 0.0
    assert result_low.svi_score == 0.0
    assert result_low.risk_tier == RiskTier.LOW


def test_calculate_svi_rejects_non_evidence_bundle_input():
    with pytest.raises(SviError) as exc_info:
        calculate_svi({"case_id": "not-a-real-bundle"}, ml_scorer=NullMLScorer())
    assert exc_info.value.code == "INVALID_INPUT"


def test_boundary_scores_map_to_expected_tiers():
    """Sweep the eight CONTRACTS.md boundary values through the public
    risk_tier_for_score function used by the full pipeline (redundant
    with Step 5's own boundary tests by design -- this is Member 3's
    end-to-end guarantee, not a duplicate of unit coverage)."""
    expected = [
        (0.0, RiskTier.LOW),
        (24.0, RiskTier.LOW),
        (25.0, RiskTier.MODERATE),
        (49.0, RiskTier.MODERATE),
        (50.0, RiskTier.HIGH),
        (79.0, RiskTier.HIGH),
        (80.0, RiskTier.CRITICAL),
        (100.0, RiskTier.CRITICAL),
    ]
    for score, tier in expected:
        assert risk_tier_for_score(score) == tier, f"score={score}"
        assert not math.isnan(score)
