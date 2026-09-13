"""
Layer 2 (SVI / Risk) — public adapter wiring the rule engine and ML
scorer together.

    EvidenceBundle -> calculate_svi(evidence_bundle) -> SVIResult

This module does not redesign either component:
- services/svi/rules.py's `calculate_rule_floor` is used as-is for the
  deterministic `rule_floor` and its explainable triggered rules.
- services/svi/ml/*'s `MLScorer` interface is used as-is for the
  optional `ml_score`. The default is `HeuristicMockScorer`, the
  clearly-labeled deterministic mock from
  services/svi/ml/heuristic_scorer.py -- there is still no real trained
  model.

Steps:
  1. Extract numeric features from the EvidenceBundle (features.py).
  2. Calculate the deterministic rule_floor (rules.py).
  3. Calculate ml_score via the injected/default MLScorer. A failing or
     absent scorer degrades to `ml_score = None`; it never breaks this
     path.
  4. final_score = max(rule_floor, ml_score) when ml_score exists,
     else final_score = rule_floor. A rule_floor established by a
     critical safety marker is never lowered by ml_score.
  5. Build the canonical SVIResult (svi_result.py).
  6. `explanation` lists, in order: each triggered rule's contribution
     (with its `RuleHit.description`, never invented here), the ML
     contribution (if any), a `rule_floor_applied` entry whenever the
     rule floor is what determined (or tied) the final score, a
     `final_score` summary entry, and a `risk_tier` entry. Every entry
     is structured data (feature/impact/description) -- never an
     LLM-generated paragraph.

Deterministic end-to-end whenever the ML component is deterministic
(the default `HeuristicMockScorer`, or `NullMLScorer`): the same
EvidenceBundle always produces the same SVIResult, aside from
`created_at`.

SAFETY NOTE: SVIResult is an assistive vulnerability/stress
PRIORITIZATION signal for human operators. It is NEVER a medical,
psychiatric, or forensic diagnosis, and it is not evidence of a
person's truthfulness or untruthfulness.
"""

from __future__ import annotations

from typing import Any

from services.fusion import EvidenceBundle
from services.svi.errors import INVALID_INPUT, SviError
from services.svi.features import extract_features
from services.svi.ml.base import MLScorer
from services.svi.ml.heuristic_scorer import HeuristicMockScorer
from services.svi.rules import calculate_rule_floor
from services.svi.svi_result import SVIResult, risk_tier_for_score


def _require_evidence_bundle(value: Any) -> EvidenceBundle:
    if not isinstance(value, EvidenceBundle):
        raise SviError(
            INVALID_INPUT,
            "calculate_svi requires a services.fusion.EvidenceBundle instance.",
        )
    return value


def calculate_svi(evidence_bundle: Any, *, ml_scorer: MLScorer | None = None) -> SVIResult:
    """
    Build a contract-shaped SVIResult from an EvidenceBundle.

    Raises SviError(INVALID_INPUT) for anything that is not a real
    EvidenceBundle. Never raises for merely-empty evidence (no markers,
    no sentiment, no voice_features, no transcript) -- that yields a
    valid, low-scoring SVIResult -- and never raises because the ML
    scorer is absent or fails; that degrades to `ml_score = None`.
    """
    evidence_bundle = _require_evidence_bundle(evidence_bundle)

    # 1. Extract relevant evidence.
    features = extract_features(evidence_bundle)

    # 2. Calculate deterministic rule score.
    rule_result = calculate_rule_floor(evidence_bundle)

    # 3. Calculate ML score. A failing/absent scorer must never break
    # this path -- it only removes the ML contribution.
    scorer = ml_scorer or HeuristicMockScorer()
    try:
        raw_ml_score = scorer.score(features)
    except Exception:
        raw_ml_score = None
    ml_score = None if raw_ml_score is None else round(max(0.0, min(100.0, float(raw_ml_score))), 1)

    # 4. Apply the project's safety rule: an ML score never lowers a
    # rule-established floor.
    final_score = max(rule_result.rule_floor, ml_score) if ml_score is not None else rule_result.rule_floor
    final_score = round(max(0.0, min(100.0, final_score)), 1)
    risk_tier = risk_tier_for_score(final_score)

    # 6. Explainability: rule contributions (with their human-readable
    # descriptions from rules.py, never invented here), the ML
    # contribution, the reason for any rule floor, the final score, and
    # the resulting risk tier -- all structured data, never an
    # LLM-generated paragraph (Step 6 checkpoint).
    explanation: list[dict[str, float | str]] = [
        {"feature": hit.code, "impact": hit.impact, "description": hit.description}
        for hit in rule_result.triggered_rules
    ]
    if ml_score is not None:
        explanation.append(
            {
                "feature": "ml_score",
                "impact": ml_score,
                "description": "Lightweight ML scorer contribution (heuristic mock unless a trained model is configured).",
            }
        )

    floor_governed = bool(rule_result.triggered_rules) and rule_result.rule_floor >= (
        ml_score if ml_score is not None else 0.0
    )
    if floor_governed:
        explanation.append(
            {
                "feature": "rule_floor_applied",
                "impact": rule_result.rule_floor,
                "description": "Deterministic rule floor determined (or tied) the final score; an ML score can never lower it.",
            }
        )

    explanation.append(
        {
            "feature": "final_score",
            "impact": final_score,
            "description": "Final SVI score: max(rule_floor, ml_score) per CONTRACTS.md 6.3.",
        }
    )
    explanation.append(
        {
            "feature": "risk_tier",
            "impact": final_score,
            "description": f"Mapped to {risk_tier.value} risk tier per CONTRACTS.md 6.3 band for this score.",
        }
    )

    # 5. Produce the canonical SVIResult.
    return SVIResult(
        case_id=evidence_bundle.case_id,
        svi_score=final_score,
        risk_tier=risk_tier,
        rule_floor=rule_result.rule_floor,
        ml_score=ml_score,
        explanation=explanation,
    )
