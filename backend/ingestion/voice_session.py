"""
Layer 0 — voice call/session identification + audio chunk buffering.

Conceptual flow this module supports:

    Twilio / Exotel -> /voice/incoming -> call/session identification
        -> audio stream or chunks -> buffering -> Input Envelope -> Layer 1

No STT, no voice-feature extraction, no stress/pitch/jitter analysis here.
This module only tracks "which call maps to which case" and buffers
references to incoming audio chunks so they can be assembled/consumed by
Layer 1. If Redis is later introduced, VoiceSessionStore can be
re-implemented against it without changing the API layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.ingestion.input_envelope import AudioReference


@dataclass
class VoiceSession:
    call_id: str
    case_id: str
    chunks: list[AudioReference] = field(default_factory=list)


class VoiceSessionStore(ABC):
    @abstractmethod
    def start_session(self, call_id: str, case_id: str) -> VoiceSession:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, call_id: str) -> VoiceSession | None:
        raise NotImplementedError

    @abstractmethod
    def buffer_chunk(self, call_id: str, chunk: AudioReference) -> None:
        raise NotImplementedError


class InMemoryVoiceSessionStore(VoiceSessionStore):
    """
    Simple in-memory buffer, adequate for a single-process dev deployment.
    A RedisVoiceSessionStore can later provide the same interface for a
    multi-process/production deployment.
    """

    def __init__(self):
        self._sessions: dict[str, VoiceSession] = {}

    def start_session(self, call_id: str, case_id: str) -> VoiceSession:
        session = VoiceSession(call_id=call_id, case_id=case_id)
        self._sessions[call_id] = session
        return session

    def get_session(self, call_id: str) -> VoiceSession | None:
        return self._sessions.get(call_id)

    def buffer_chunk(self, call_id: str, chunk: AudioReference) -> None:
        session = self._sessions.get(call_id)
        if session is None:
            raise KeyError(f"Unknown call_id '{call_id}'; call /voice/incoming first.")
        session.chunks.append(chunk)
