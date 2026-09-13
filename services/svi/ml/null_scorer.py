"""
Layer 2 (SVI / Risk) — default "no model configured" scorer.

Always returns `None`, which is the safe, expected way for a caller to
end up with `ml_score = None` and fall back to the deterministic
`rule_floor` alone. This is what "missing ML model" looks like in this
project -- not an error condition, not a raised exception.

No dependencies, no I/O, nothing to load.
"""

from __future__ import annotations

from typing import Mapping

from services.svi.ml.base import MLScorer


class NullMLScorer(MLScorer):
    """Default scorer used whenever no other MLScorer is supplied."""

    def score(self, features: Mapping[str, float]) -> float | None:
        return None
