"""
Layer 0 — voice call/session identification.

Conceptual flow this module supports:

    Twilio / Exotel -> /voice/incoming -> call/session identification
        -> audio stream or chunks -> Input Envelope -> Layer 1

No STT, no voice-feature extraction, no stress/pitch/jitter analysis here.
This module only tracks "which active call maps to which case." Cumulative
voice evidence belongs to Layer 4B's orchestration voice session store. If
Redis is later introduced, VoiceSessionStore can be re-implemented against
it without changing the API layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VoiceSession:
    call_id: str
    case_id: str


class VoiceSessionStore(ABC):
    @abstractmethod
    def start_session(self, call_id: str, case_id: str) -> VoiceSession:
        raise NotImplementedError

    @abstractmethod
    def get_session(self, call_id: str) -> VoiceSession | None:
        raise NotImplementedError

    @abstractmethod
    def end_session(self, call_id: str) -> None:
        raise NotImplementedError


class InMemoryVoiceSessionStore(VoiceSessionStore):
    """
    Simple in-memory identity store, adequate for a single-process dev deployment.
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

    def end_session(self, call_id: str) -> None:
        self._sessions.pop(call_id, None)
