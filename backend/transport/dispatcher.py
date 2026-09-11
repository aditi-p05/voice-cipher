"""
Layer 0 — async handoff abstraction.

Layer 0 hands validated envelopes to downstream processing through this
thin interface. The in-memory implementation is sufficient for
development/testing; a RedisStreamDispatcher (or any other backend) can be
substituted later without changing the API layer, since callers only ever
depend on the InputDispatcher interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.ingestion.input_envelope import InputEnvelope


@dataclass
class DispatchResult:
    accepted: bool
    input_id: str
    case_id: str
    queue_name: str


class InputDispatcher(ABC):
    """Interface all dispatch backends must implement."""

    @abstractmethod
    def publish(self, envelope: InputEnvelope) -> DispatchResult:
        raise NotImplementedError


class InMemoryDispatcher(InputDispatcher):
    """
    Development/default dispatcher. Holds published envelopes in a local
    list so tests and local runs can inspect what would have been sent
    downstream. Not durable, not distributed, and not intended for
    production use once Redis (or similar) is wired in.
    """

    def __init__(self, queue_name: str = "layer0.intake"):
        self.queue_name = queue_name
        self._published: list[InputEnvelope] = []

    def publish(self, envelope: InputEnvelope) -> DispatchResult:
        self._published.append(envelope)
        return DispatchResult(
            accepted=True,
            input_id=envelope.input_id,
            case_id=envelope.case_id,
            queue_name=self.queue_name,
        )

    @property
    def published(self) -> list[InputEnvelope]:
        return list(self._published)


class RedisStreamDispatcher(InputDispatcher):
    """
    Placeholder for a future Redis Streams-backed dispatcher.

    Intentionally NOT implemented in Layer 0 (no Redis infra exists yet in
    this project). Kept here only to document the swap-in point so the API
    layer never has to change: construct whichever InputDispatcher is
    configured and call `.publish(envelope)`.
    """

    def __init__(self, redis_client, stream_key: str = "layer0:intake"):
        self._redis = redis_client
        self._stream_key = stream_key

    def publish(self, envelope: InputEnvelope) -> DispatchResult:  # pragma: no cover
        raise NotImplementedError(
            "RedisStreamDispatcher is a placeholder for a future task; "
            "Redis is not configured in this project yet."
        )
