"""
Layer 2 (SVI / Risk) — pluggable ML scoring interface.

Two small, separate interfaces by design:

- `ModelLoader` obtains whatever a scorer needs to run (weights, a
  serialized model, a registry handle, a file path, ...). This is the
  only place I/O (or anything slow/fallible) is expected to happen.
- `MLScorer` turns already-loaded state + a feature dict into a score.
  `score()` itself must be cheap, CPU-only, and side-effect free.

Keeping these separate means model loading can fail, be slow, or be
swapped out (mock weights today, a real trained artifact tomorrow)
without ever touching the scoring math, and vice versa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ModelLoader(Protocol):
    """Obtains whatever a scorer needs to run. May do I/O; may fail."""

    def load(self) -> Any:
        ...


class MLScorer(ABC):
    """
    Layer 2's pluggable ML scoring interface.

    Implementations MUST:
    - return a float in the project's expected range [0, 100], or
      `None` if no score is available (no model configured, or a
      failure occurred);
    - never let a raw/unexpected exception escape `score()` -- catch
      internal failures (bad/missing loader output, malformed
      features, arithmetic errors, ...) and return `None` instead.

    The deterministic rule engine (services/svi/rules.py) is the
    safety-critical path and must never depend on an MLScorer
    succeeding; `None` is always a safe, expected result here, not an
    error condition.
    """

    @abstractmethod
    def score(self, features: Mapping[str, float]) -> float | None:
        """Score a flat dict of numeric features into [0, 100], or
        return None if unavailable. Feature extraction from
        EvidenceBundle is a separate concern, owned by whatever calls
        this scorer -- not by MLScorer implementations themselves."""
        raise NotImplementedError
