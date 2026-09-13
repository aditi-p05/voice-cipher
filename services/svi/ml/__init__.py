"""
Layer 2 (SVI / Risk) — pluggable ML scoring components.

    from services.svi.ml import MLScorer, ModelLoader, NullMLScorer, HeuristicMockScorer

`NullMLScorer` is the safe default (no model configured -> always
`None`). `HeuristicMockScorer` is a clearly-labeled, deterministic
mock -- NOT a trained model -- usable as a stand-in until a real
trained artifact exists.
"""

from services.svi.ml.base import MLScorer, ModelLoader
from services.svi.ml.heuristic_scorer import DefaultWeightsLoader, HeuristicMockScorer
from services.svi.ml.null_scorer import NullMLScorer

__all__ = [
    "MLScorer",
    "ModelLoader",
    "NullMLScorer",
    "HeuristicMockScorer",
    "DefaultWeightsLoader",
]
