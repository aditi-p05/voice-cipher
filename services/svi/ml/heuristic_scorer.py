"""
Layer 2 (SVI / Risk) — deterministic mock ML scorer.

*** NOT A TRAINED MODEL ***

There is no real trained artifact for Layer 2 (no labeled real, or
ethically sourced and realistically distributed, training data exists
in this project). Per the Layer 2 brief -- "if there is insufficient
real training data, use synthetic/demo data, clearly label it, provide
deterministic fallback scoring, keep the architecture ready for a real
trained model later" -- this scorer is a small, clearly-labeled,
hand-written, deterministic stand-in: a fixed linear combination of
numeric features, clamped into [0, 100]. It exists so the "hybrid
deterministic + ML" architecture and its call shape can be exercised
end-to-end (including in CI), not to make a real risk prediction.

Lightweight and CPU-friendly on purpose: pure Python arithmetic, no
NumPy/scikit-learn/XGBoost/etc. and no model download. Introduce a real
ML dependency only when there is an actual trained artifact to load.

Model loading is kept separate from scoring: `DefaultWeightsLoader`
supplies the weight vector (the closest thing this scorer has to a
"model artifact"); `HeuristicMockScorer.score()` only does the (cheap)
linear combination once weights are loaded and cached. Swapping in a
real trained model later means writing a new `MLScorer` + `ModelLoader`
pair behind the same interface (services/svi/ml/base.py) -- nothing
else in Layer 2 needs to change.

Before any real deployment, replace this scorer (and/or supply a real
`ModelLoader`) with one trained on properly governed, real (or
realistically distributed and ethically sourced) labeled data.
"""

from __future__ import annotations

from typing import Mapping

from services.svi.ml.base import MLScorer, ModelLoader

# NOT LEARNED FROM DATA. Hand-picked to produce a plausible-looking demo
# signal only: higher marker counts/confidence/co-occurrence, more
# negative sentiment valence, higher arousal, and more voice-stress
# indicators push the mock score up. Unknown features are ignored;
# missing features default to 0.0 (see HeuristicMockScorer.score).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "marker_count": 4.0,
    "max_marker_confidence": 25.0,
    "distinct_marker_types": 6.0,
    "sentiment_valence": -15.0,
    "sentiment_arousal": 15.0,
    "voice_pitch_stdev": 0.1,
    "voice_jitter": 50.0,
    "voice_shimmer": 50.0,
    "voice_pause_frequency": 1.0,
}
_BIAS = 0.0


class DefaultWeightsLoader(ModelLoader):
    """Loads the built-in mock weight vector. No file I/O, no network."""

    def load(self) -> dict[str, float]:
        return dict(_DEFAULT_WEIGHTS)


class HeuristicMockScorer(MLScorer):
    """
    Deterministic, CPU-only mock scorer.

    Loading is separate from scoring: `loader.load()` is called at most
    once (lazily, on first `score()` call) and cached; `score()` itself
    performs no I/O. A failing or malformed loader degrades to
    `score() -> None` rather than raising.
    """

    def __init__(self, loader: ModelLoader | None = None):
        self._loader = loader or DefaultWeightsLoader()
        self._weights: dict[str, float] | None = None

    def _ensure_loaded(self) -> dict[str, float]:
        if self._weights is None:
            loaded = self._loader.load()
            self._weights = dict(loaded) if loaded else {}
        return self._weights

    def score(self, features: Mapping[str, float]) -> float | None:
        try:
            weights = self._ensure_loaded()
            raw = _BIAS + sum(weights.get(name, 0.0) * float(value) for name, value in features.items())
        except Exception:
            # A loader or arithmetic failure must never raise out of
            # score() -- degrade to "no ML score available" instead.
            return None
        return max(0.0, min(100.0, raw))
