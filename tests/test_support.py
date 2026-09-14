import pytest

from services.support.errors import SupportError
from services.support.support_service import decide_support


def evidence(sentiment=None, **overrides):
    result = {
        "schema_version": "1.0.0",
        "case_id": "CASE-TEST",
        "source_input_ids": ["IN-1"],
        "transcript": None,
        "markers": [],
        "sentiment": sentiment,
        "voice_features": None,
        "pii_redacted": True,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    result.update(overrides)
    return result


def svi(tier="LOW", **overrides):
    result = {
        "schema_version": "1.0.0",
        "case_id": "CASE-TEST",
        "svi_score": 10,
        "risk_tier": tier,
        "rule_floor": 0,
        "ml_score": 10,
        "explanation": [],
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("tier", ["LOW", "MODERATE"])
def test_low_and_moderate_risk_support_is_enabled(tier):
    decision = decide_support(evidence(sentiment={"valence": -0.5, "arousal": 0.3}), svi(tier))
    assert decision["support_enabled"] is True
    assert decision["mode"] == "enabled"
    assert decision["preset"] is not None
    assert decision["disclaimer"]


def test_high_risk_support_is_restricted():
    decision = decide_support(evidence(sentiment={"valence": -0.8, "arousal": 0.9}), svi("HIGH", svi_score=60, rule_floor=0, ml_score=60))
    assert decision["support_enabled"] is False
    assert decision["mode"] == "restricted"
    assert decision["preset"] is None


def test_critical_risk_support_is_disabled():
    decision = decide_support(
        evidence(sentiment={"valence": -0.9, "arousal": 0.95}),
        svi("CRITICAL", svi_score=90, rule_floor=80, ml_score=90),
    )
    assert decision["support_enabled"] is False
    assert decision["mode"] == "disabled"
    assert decision["preset"] is None
    assert "priority" in decision["reason"].lower()


def test_missing_sentiment_falls_back_to_neutral_checkin_not_an_error():
    decision = decide_support(evidence(sentiment=None), svi("LOW"))
    assert decision["support_enabled"] is True
    assert decision["preset"]["preset_id"] == "neutral_checkin"


@pytest.mark.parametrize(
    "valence,arousal,expected_preset",
    [
        (-0.7, 0.8, "grounding_calm"),
        (-0.5, 0.2, "gentle_regulation"),
        (-0.1, 0.1, "social_support_guidance"),
        (0.5, 0.1, "neutral_checkin"),
    ],
)
def test_preset_selection_by_valence_arousal(valence, arousal, expected_preset):
    decision = decide_support(evidence(sentiment={"valence": valence, "arousal": arousal}), svi("MODERATE"))
    assert decision["preset"]["preset_id"] == expected_preset


def test_malformed_evidence_bundle_raises_support_error():
    with pytest.raises(SupportError):
        decide_support({"case_id": "CASE-TEST"}, svi())


def test_missing_svi_fields_raises_support_error():
    with pytest.raises(SupportError):
        decide_support(evidence(), {"case_id": "CASE-TEST"})


def test_mismatched_case_ids_raises_support_error():
    with pytest.raises(SupportError):
        decide_support(evidence(case_id="CASE-A"), svi(case_id="CASE-B") | {"case_id": "CASE-B"})


def test_invalid_risk_tier_raises_support_error():
    with pytest.raises(SupportError):
        decide_support(evidence(), svi("NOT_A_TIER"))


def test_no_therapeutic_claims_in_reason():
    # The `reason` field must never make an affirmative therapeutic claim.
    # (The disclaimer is expected to *name and deny* these words -- "not
    # medical treatment, therapy, or a diagnosis" -- which is the required
    # safe framing, not a violation, so it is intentionally not checked here.)
    decision = decide_support(evidence(sentiment={"valence": -0.5, "arousal": 0.3}), svi("LOW"))
    forbidden = {"treat", "treatment", "therapy", "diagnos", "cure"}
    assert not any(word in decision["reason"].lower() for word in forbidden)


def test_disclaimer_explicitly_denies_clinical_framing():
    decision = decide_support(evidence(sentiment={"valence": -0.5, "arousal": 0.3}), svi("LOW"))
    disclaimer = decision["disclaimer"].lower()
    assert "not medical treatment" in disclaimer
    assert "therapy" in disclaimer and "not" in disclaimer
