"""
Layer 0 — dependency wiring.

Process-local singletons for development. Swapping InMemoryDispatcher ->
RedisStreamDispatcher, or InMemoryCaseStore -> a real DB-backed store,
only requires changing this module.
"""

from __future__ import annotations

from backend.ingestion.case_store import InMemoryCaseStore
from backend.ingestion.intake_service import IntakeService
from backend.ingestion.voice_session import (
    InMemoryVoiceSessionStore as IngestionVoiceSessionStore,
)
from backend.orchestration.voice_session import (
    InMemoryVoiceSessionStore as OrchestrationVoiceSessionStore,
)
from backend.transport.dispatcher import InMemoryDispatcher
from backend.cases.store import InMemoryCaseReadStore
from backend.cases.service import CaseIntegrationService

_case_store = InMemoryCaseStore()
_dispatcher = InMemoryDispatcher()
_ingestion_voice_sessions = IngestionVoiceSessionStore()
_orchestration_voice_sessions = OrchestrationVoiceSessionStore()
_intake_service = IntakeService(case_store=_case_store, dispatcher=_dispatcher)
_case_read_store = InMemoryCaseReadStore()
_case_integration_service = CaseIntegrationService(_case_read_store)


def get_intake_service() -> IntakeService:
    return _intake_service


def get_voice_session_store() -> IngestionVoiceSessionStore:
    """Provide Layer 0 call-to-case voice session identity state."""
    return _ingestion_voice_sessions


def get_orchestration_voice_sessions() -> OrchestrationVoiceSessionStore:
    """Provide Layer 4B cumulative evidence state for voice calls."""
    return _orchestration_voice_sessions


def get_dispatcher() -> InMemoryDispatcher:
    return _dispatcher


def get_case_store() -> InMemoryCaseStore:
    return _case_store

def get_case_integration_service() -> CaseIntegrationService:
    return _case_integration_service
