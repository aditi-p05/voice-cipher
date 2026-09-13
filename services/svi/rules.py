"""
Layer 2 (SVI / Risk) — deterministic safety rule engine.

Goal: EvidenceBundle -> rule-based risk indicators (`rule_floor`).

Deliberately marker/keyword-type based (mirrors the extractor-level
design already used in services/fusion/markers/marker_extractor.py) so
behavior is predictable, auditable, and testable -- not a black box.
This module does not use ML, does not call RAG, and does not touch
dashboard/orchestration concerns.

Every triggered rule carries a machine-readable `code` plus a short
human-readable `description`, generated directly from the same table
used to compute `rule_floor` -- never invented after the fact.

CRITICAL-SAFETY DESIGN: an explicit critical marker (self-harm, weapon,
immediate danger, explicit threat, severe violence) can establish a
`rule_floor` on its own, high enough to land in the CRITICAL band (>=80)
at full confidence. Per the project design, this floor is a MINIMUM --
nothing computed later (e.g. an ML score, not implemented yet) is
allowed to pull the final score below it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.fusion import EvidenceBundle, Marker


@dataclass(frozen=True)
class RuleHit:
    """One triggered deterministic rule."""

    code: str
    description: str
    impact: float
    marker_type: str | None = None


@dataclass(frozen=True)
class RuleEngineResult:
    """Output of the deterministic rule engine."""

    rule_floor: float
    triggered_rules: list[RuleHit] = field(default_factory=list)


# marker_type -> (machine-readable code, human-readable description,
# base floor contribution at full (1.0) marker confidence). These five
# are the safety-critical markers called out in the Layer 2 brief:
# explicit threat, self-harm, weapon, immediate danger, severe violence.
_CRITICAL_MARKER_RULES: dict[str, tuple[str, str, float]] = {
    "self_harm": ("CRITICAL_SELF_HARM", "Explicit self-harm marker present", 90.0),
    "weapon": ("CRITICAL_WEAPON", "Weapon marker present", 88.0),
    "immediate_danger": ("CRITICAL_IMMEDIATE_DANGER", "Immediate danger marker present", 85.0),
    "threat": ("CRITICAL_EXPLICIT_THREAT", "Explicit threat marker present", 82.0),
    "violence": ("CRITICAL_SEVERE_VIOLENCE", "Severe violence marker present", 80.0),
}

# Supporting vulnerability indicators. Still explainable and still able
# to move rule_floor into MODERATE/HIGH territory, but not treated as an
# unconditional critical-safety floor the way the markers above are.
_SUPPORTING_MARKER_RULES: dict[str, tuple[str, str, float]] = {
    "coercion": ("SUPPORTING_COERCION", "Coercion marker present", 45.0),
    "retaliation": ("SUPPORTING_RETALIATION", "Retaliation marker present", 40.0),
    "unsafe": ("SUPPORTING_UNSAFE_ENVIRONMENT", "Unsafe-environment marker present", 35.0),
    "fear": ("SUPPORTING_FEAR", "Fear marker present", 30.0),
}

# A small, capped bonus when multiple distinct concerning marker types
# co-occur -- corroborating signals across independent cues are more
# trustworthy than a single isolated cue, but this must never become the
# dominant driver of rule_floor.
_COOCCURRENCE_CODE = "MULTIPLE_CONCERNING_MARKERS"
_COOCCURRENCE_DESCRIPTION = "Multiple distinct concerning marker types corroborate each other"
_COOCCURRENCE_STEP = 3.0
_MAX_COOCCURRENCE_BONUS = 10.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def calculate_rule_floor(evidence_bundle: EvidenceBundle) -> RuleEngineResult:
    """
    Compute the deterministic `rule_floor` and its triggered rules from
    `evidence_bundle.markers`.

    Unknown/unrecognized marker types trigger no rule (ignored, not
    rejected), so an unexpected marker vocabulary from Fusion never
    breaks this module. Missing/empty markers (or any other absent
    optional evidence on the bundle) simply yields no triggered rules
    and a rule_floor of 0.0 -- this module never raises for that case.

    Deterministic and side-effect free: the same markers always produce
    the same result, which is required for repeatable, auditable risk
    scoring.
    """
    markers: list[Marker] = list(getattr(evidence_bundle, "markers", None) or [])

    best_hit_by_type: dict[str, RuleHit] = {}
    for marker in markers:
        rule = _CRITICAL_MARKER_RULES.get(marker.type) or _SUPPORTING_MARKER_RULES.get(marker.type)
        if rule is None:
            continue
        code, description, base_floor = rule
        impact = round(base_floor * _clamp01(marker.confidence), 1)
        existing = best_hit_by_type.get(marker.type)
        if existing is None or impact > existing.impact:
            best_hit_by_type[marker.type] = RuleHit(
                code=code, description=description, impact=impact, marker_type=marker.type
            )

    if not best_hit_by_type:
        return RuleEngineResult(rule_floor=0.0, triggered_rules=[])

    ordered_hits = sorted(best_hit_by_type.values(), key=lambda hit: -hit.impact)
    primary_hit = ordered_hits[0]

    distinct_types = len(ordered_hits)
    cooccurrence_bonus = (
        min(_MAX_COOCCURRENCE_BONUS, _COOCCURRENCE_STEP * (distinct_types - 1)) if distinct_types > 1 else 0.0
    )

    rule_floor = round(max(0.0, min(100.0, primary_hit.impact + cooccurrence_bonus)), 1)

    triggered_rules = list(ordered_hits)
    if cooccurrence_bonus:
        triggered_rules.append(
            RuleHit(
                code=_COOCCURRENCE_CODE,
                description=_COOCCURRENCE_DESCRIPTION,
                impact=round(cooccurrence_bonus, 1),
            )
        )

    return RuleEngineResult(rule_floor=rule_floor, triggered_rules=triggered_rules)
