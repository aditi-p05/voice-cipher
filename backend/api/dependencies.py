"""
Layer 0 — dependency wiring.

Process-local singletons for development. Swapping InMemoryDispatcher ->
RedisStreamDispatcher, or InMemoryCaseStore -> a real DB-backed store,
only requires changing this module.
"""

from __future__ import annotations

from backend.ingestion.case_store import InMemoryCaseStore
from backend.ingestion.intake_service import IntakeService
from backend.ingestion.voice_session import InMemoryVoiceSessionStore
from backend.transport.dispatcher import InMemoryDispatcher

_case_store = InMemoryCaseStore()
_dispatcher = InMemoryDispatcher()
_voice_sessions = InMemoryVoiceSessionStore()
_intake_service = IntakeService(case_store=_case_store, dispatcher=_dispatcher)


def get_intake_service() -> IntakeService:
    return _intake_service


def get_voice_session_store() -> InMemoryVoiceSessionStore:
    return _voice_sessions


def get_dispatcher() -> InMemoryDispatcher:
    return _dispatcher


def get_case_store() -> InMemoryCaseStore:
    return _case_store
