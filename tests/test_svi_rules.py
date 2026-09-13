"""
Focused unit tests for the deterministic SVI rule engine
(services/svi/rules.py): EvidenceBundle -> rule_floor.

Scope: rule engine only. No ML, no RAG, no dashboard/orchestration.
"""

from __future__ import annotations

from services.fusion import EvidenceBundle, Marker
from services.svi import RiskTier, risk_tier_for_score
from services.svi.rules import calculate_rule_floor


def _bundle(markers=None, **overrides) -> EvidenceBundle:
    kwargs = {
        "case_id": "CASE-RULES0001",
        "source_input_ids": ["IN-1"],
        "markers": markers or [],
        "pii_redacted": True,
    }
    kwargs.update(overrides)
    return EvidenceBundle(**kwargs)


# ---------------------------------------------------------------------------
# 1. No risk indicators
# ---------------------------------------------------------------------------
def test_no_markers_produces_zero_rule_floor_and_no_triggered_rules():
    result = calculate_rule_floor(_bundle(markers=[]))
    assert result.rule_floor == 0.0
    assert result.triggered_rules == []


# ---------------------------------------------------------------------------
# 2. One vulnerability indicator
# ---------------------------------------------------------------------------
def test_single_supporting_marker_produces_matching_rule_floor():
    bundle = _bundle(markers=[Marker(type="fear", value="i am scared", confidence=0.7)])
    result = calculate_rule_floor(bundle)

    assert result.rule_floor == 21.0  # 30.0 base * 0.7 confidence
    assert len(result.triggered_rules) == 1
    assert result.triggered_rules[0].code == "SUPPORTING_FEAR"
    assert result.triggered_rules[0].impact == 21.0


# ---------------------------------------------------------------------------
# 3. Multiple indicators
# ---------------------------------------------------------------------------
def test_multiple_distinct_markers_combine_with_a_capped_cooccurrence_bonus():
    bundle = _bundle(
        markers=[
            Marker(type="unsafe", value="not safe", confidence=1.0),  # impact 35.0
            Marker(type="fear", value="i am scared", confidence=0.5),  # impact 15.0
        ]
    )
    result = calculate_rule_floor(bundle)

    # primary (35.0) + cooccurrence bonus (3.0 for 2 distinct types) = 38.0
    assert result.rule_floor == 38.0
    codes = {hit.code for hit in result.triggered_rules}
    assert codes == {"SUPPORTING_UNSAFE_ENVIRONMENT", "SUPPORTING_FEAR", "MULTIPLE_CONCERNING_MARKERS"}


# ---------------------------------------------------------------------------
# 4. Explicit critical indicator
# ---------------------------------------------------------------------------
def test_explicit_critical_marker_establishes_a_critical_rule_floor():
    bundle = _bundle(markers=[Marker(type="self_harm", value="kill myself", confidence=1.0)])
    result = calculate_rule_floor(bundle)

    assert result.rule_floor == 90.0
    assert len(result.triggered_rules) == 1
    assert result.triggered_rules[0].code == "CRITICAL_SELF_HARM"
    # The rule floor alone (no ML score involved) already lands in CRITICAL.
    assert risk_tier_for_score(result.rule_floor) is RiskTier.CRITICAL


# ---------------------------------------------------------------------------
# 5. Missing/empty optional evidence
# ---------------------------------------------------------------------------
def test_missing_optional_evidence_does_not_crash_and_yields_zero_floor():
    bundle = _bundle(markers=[], transcript=None, sentiment=None, voice_features=None)
    result = calculate_rule_floor(bundle)

    assert result.rule_floor == 0.0
    assert result.triggered_rules == []
