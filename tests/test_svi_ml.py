"""
Focused unit tests for Layer 2's ML scoring component
(services/svi/ml/): interface, safe fallback, and the deterministic
mock scorer. No real model is trained or downloaded anywhere here.
"""

from __future__ import annotations

import pytest

from services.svi.ml import (
    DefaultWeightsLoader,
    HeuristicMockScorer,
    MLScorer,
    ModelLoader,
    NullMLScorer,
)


# ---------------------------------------------------------------------------
# NullMLScorer: the safe "no model configured" default
# ---------------------------------------------------------------------------
def test_null_scorer_always_returns_none():
    scorer = NullMLScorer()
    assert scorer.score({}) is None
    assert scorer.score({"marker_count": 5.0, "sentiment_valence": -0.9}) is None


def test_null_scorer_conforms_to_mlscorer_interface():
    assert isinstance(NullMLScorer(), MLScorer)


# ---------------------------------------------------------------------------
# A fake/mock scorer conforming to the interface, to prove pluggability
# ---------------------------------------------------------------------------
class _FakeFixedScorer(MLScorer):
    """Minimal fake used only to prove the interface is easy to satisfy."""

    def __init__(self, fixed_value: float | None):
        self._fixed_value = fixed_value

    def score(self, features):
        return self._fixed_value


def test_fake_scorer_can_be_used_interchangeably_via_the_interface():
    def run(scorer: MLScorer, features: dict) -> float | None:
        return scorer.score(features)

    assert run(_FakeFixedScorer(42.0), {"marker_count": 1.0}) == 42.0
    assert run(NullMLScorer(), {"marker_count": 1.0}) is None


# ---------------------------------------------------------------------------
# HeuristicMockScorer: deterministic fallback (NOT a trained model)
# ---------------------------------------------------------------------------
def test_heuristic_mock_scorer_returns_value_in_expected_range():
    scorer = HeuristicMockScorer()
    result = scorer.score(
        {
            "marker_count": 2.0,
            "max_marker_confidence": 0.8,
            "distinct_marker_types": 2.0,
            "sentiment_valence": -0.6,
            "sentiment_arousal": 0.7,
        }
    )
    assert result is not None
    assert 0.0 <= result <= 100.0


def test_heuristic_mock_scorer_clamps_to_upper_bound():
    scorer = HeuristicMockScorer()
    # Deliberately extreme feature values that would blow past 100 under
    # the raw linear combination.
    result = scorer.score({"marker_count": 100.0, "voice_jitter": 10.0})
    assert result == 100.0


def test_heuristic_mock_scorer_clamps_to_lower_bound():
    scorer = HeuristicMockScorer()
    # sentiment_valence has a negative weight, so a strongly positive
    # valence should push the raw score below zero -> clamped to 0.0.
    result = scorer.score({"sentiment_valence": 1.0})
    assert result == 0.0


def test_heuristic_mock_scorer_handles_missing_features_gracefully():
    scorer = HeuristicMockScorer()
    assert scorer.score({}) == 0.0  # bias is 0.0, no features -> 0.0, no crash


def test_heuristic_mock_scorer_is_deterministic():
    scorer = HeuristicMockScorer()
    features = {"marker_count": 3.0, "sentiment_arousal": 0.5}
    assert scorer.score(features) == scorer.score(features)


def test_heuristic_mock_scorer_uses_injected_loader_not_the_default():
    class _FakeLoader(ModelLoader):
        def load(self):
            return {"marker_count": 1000.0}  # wildly different from defaults

    scorer = HeuristicMockScorer(loader=_FakeLoader())
    # A single marker with the fake loader's huge weight should saturate
    # immediately, proving the injected loader (not DefaultWeightsLoader)
    # was actually used.
    assert scorer.score({"marker_count": 1.0}) == 100.0


def test_heuristic_mock_scorer_loader_failure_returns_none_not_raises():
    class _FailingLoader(ModelLoader):
        def load(self):
            raise RuntimeError("weights file missing")

    scorer = HeuristicMockScorer(loader=_FailingLoader())
    assert scorer.score({"marker_count": 1.0}) is None


def test_heuristic_mock_scorer_loader_is_called_at_most_once():
    class _CountingLoader(ModelLoader):
        def __init__(self):
            self.calls = 0

        def load(self):
            self.calls += 1
            return {"marker_count": 1.0}

    loader = _CountingLoader()
    scorer = HeuristicMockScorer(loader=loader)
    scorer.score({"marker_count": 1.0})
    scorer.score({"marker_count": 2.0})
    assert loader.calls == 1


def test_default_weights_loader_returns_plain_dict():
    weights = DefaultWeightsLoader().load()
    assert isinstance(weights, dict)
    assert "marker_count" in weights


@pytest.mark.parametrize("bad_features", [{"marker_count": "not-a-number"}])
def test_heuristic_mock_scorer_malformed_features_return_none_not_raise(bad_features):
    scorer = HeuristicMockScorer()
    assert scorer.score(bad_features) is None
