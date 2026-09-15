"""
backend/transport/dispatcher.py

*** STUB — this did NOT ship in the zip you gave me. ***

Both `backend/ingestion/intake_service.py` and `backend/api/dependencies.py`
import `InputDispatcher` / `DispatchResult` / `InMemoryDispatcher`
unconditionally -- without this module the FastAPI app (`backend.main`)
cannot be imported at all, so none of the routes are actually
reachable over HTTP today.

Layer 0's own docstrings describe this as the boundary that would, in
production, publish an InputEnvelope onto a real queue (the comment in
dependencies.py literally says "Swapping InMemoryDispatcher ->
RedisStreamDispatcher ... only requires changing this module"). That's
Member 1's territory. This stub is the simplest thing that satisfies
the contract every caller already assumes: an ABC with `.publish()`,
and an in-memory implementation that always accepts and records what
it published (useful for tests/demo, e.g. inspecting what got
dispatched without a real broker running).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock

from backend.ingestion.input_envelope import InputEnvelope


@dataclass
class DispatchResult:
    accepted: bool
    queue_name: str
    reason: str = ""


class InputDispatcher(ABC):
    @abstractmethod
    def publish(self, envelope: InputEnvelope) -> DispatchResult:
        ...


class InMemoryDispatcher(InputDispatcher):
    """Development-only in-memory dispatcher. Always accepts; keeps a
    log of published envelopes so tests/demos can inspect what went
    out without a real broker. Replace with a real
    Redis/Kafka/SQS-backed dispatcher for anything beyond a prototype
    -- see this module's own docstring."""

    def __init__(self, queue_name: str = "default-intake") -> None:
        self._queue_name = queue_name
        self._published: list[InputEnvelope] = []
        self._lock = Lock()

    def publish(self, envelope: InputEnvelope) -> DispatchResult:
        with self._lock:
            self._published.append(envelope)
        return DispatchResult(accepted=True, queue_name=self._queue_name)

    @property
    def published(self) -> list[InputEnvelope]:
        """Test/demo helper -- not part of the InputDispatcher contract."""
        with self._lock:
            return list(self._published)
