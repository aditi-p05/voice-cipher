"""
Layer 4B (Orchestration) — multi-chunk voice call session accumulation.

THE PROBLEM
-----------
/voice/chunk arrives repeatedly for a single call_id (Layer 0,
CONTRACTS.md section 9). Each chunk's own InputEnvelope carries exactly
ONE audio reference -- Layer 0 does not, and should not, accumulate the
whole call itself (that would make Layer 0 stateful in a way CONTRACTS.md
never scoped to it). Running the pipeline on a lone chunk in isolation
has two real problems for a safety-critical helpline:

  1. Early chunks get judged without later, more informative evidence.
  2. Worse: a critical safety marker found in chunk 1 could be diluted or
     lost entirely if a later chunk's SVI run doesn't know chunk 1 ever
     happened (each chunk-as-its-own-envelope would otherwise be scored
     completely independently, with no memory of the rest of the call).

THE DECISION
------------
Accumulate every chunk's fusion output for a call into one running
in-memory session, and re-run SVI/Service/Support/Dashboard against the
MERGED cumulative evidence on every single chunk arrival -- not on a
debounced or batched schedule.

This is deliberately chosen over batching/debouncing for a
safety-critical system: with the current in-memory, local, non-network-
bound providers (mock/regex-based fusion, local vector store), the cost
of re-scoring per chunk is small. Running less often would delay exactly
the class of critical-safety detection this system exists to catch -- if
a caller discloses immediate danger in chunk 2 of 6, the operator must
not wait for chunk 6 to see it. If a production STT/ML provider later
makes real-time per-chunk re-scoring too expensive, that tradeoff should
be revisited with real latency numbers in hand -- not loosened pre-
emptively based on a guess.

Fusion still runs once per individual chunk (Member 2's contract is
untouched: one InputEnvelope in, one EvidenceBundle out). This module's
job is purely to MERGE the resulting per-chunk EvidenceBundles
(evidence_merge.py) before handing the cumulative picture to SVI onward
-- it never re-implements or duplicates fusion logic itself.

STORAGE
-------
In-memory only, exactly like Layer 0's own InputDispatcher and case/
session adapters (CONTRACTS.md section 14: "Current in-memory adapters
are development-only"). Whether voice sessions eventually need a
persistent store is a cross-cutting decision (case/session ownership
spans Members 1, 5, and 6) and is deliberately NOT decided unilaterally
here -- it needs the same team agreement CONTRACTS.md section 15
requires for any shared-boundary change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Optional

from backend.ingestion.input_envelope import InputEnvelope
from backend.orchestration.adapters.fusion_adapter import run_fusion
from backend.orchestration.evidence_merge import merge_evidence_bundles
from backend.orchestration.nodes import Providers
from backend.orchestration.run import run_pipeline_from_evidence
from services.fusion.evidence_bundle import EvidenceBundle


@dataclass
class CallSession:
    call_id: str
    case_id: str
    chunk_bundles: list[EvidenceBundle] = field(default_factory=list)
    chunk_errors: list[dict] = field(default_factory=list)


class InMemoryVoiceSessionStore:
    """Development in-memory store. See module docstring: a persistent
    store is a cross-cutting decision for the team, not this module."""

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}
        self._lock = Lock()

    def get_or_create(self, call_id: str, case_id: str) -> CallSession:
        with self._lock:
            session = self._sessions.get(call_id)
            if session is None:
                session = CallSession(call_id=call_id, case_id=case_id)
                self._sessions[call_id] = session
            return session

    def clear(self, call_id: str) -> None:
        with self._lock:
            self._sessions.pop(call_id, None)

    def get(self, call_id: str) -> Optional[CallSession]:
        with self._lock:
            return self._sessions.get(call_id)


_default_store = InMemoryVoiceSessionStore()


def ingest_voice_chunk(
    envelope: InputEnvelope,
    *,
    store: InMemoryVoiceSessionStore | None = None,
    providers: Providers | None = None,
) -> dict[str, Any]:
    """
    Handle one /voice/chunk envelope: fuse it individually, merge it into
    the call's running evidence, and re-run SVI/Service/Support/Dashboard
    on the cumulative picture. See module docstring for why this runs on
    every chunk rather than on a debounced schedule.
    """
    if envelope.audio is None or not envelope.audio.call_id:
        raise ValueError("Voice chunk envelope must carry audio.call_id")

    active_store = store or _default_store
    call_id = envelope.audio.call_id
    session = active_store.get_or_create(call_id, envelope.case_id)

    fusion_fn = providers.fusion_fn if providers else None
    chunk_bundle, error = run_fusion(envelope, fusion_fn=fusion_fn)
    if error:
        session.chunk_errors.append(error)
    if chunk_bundle is not None:
        session.chunk_bundles.append(chunk_bundle)

    if not session.chunk_bundles:
        # Every chunk so far has failed fusion; do not fabricate evidence.
        return {
            "schema_version": "1.0.0",
            "case_id": envelope.case_id,
            "call_id": call_id,
            "chunks_processed": 0,
            "evidence_available": False,
            "svi_result": None,
            "risk_tier": None,
            "service_recommendation": None,
            "support_decision": None,
            "requires_operator_confirmation": False,
            "errors": list(session.chunk_errors),
        }

    merged = merge_evidence_bundles(session.chunk_bundles)
    case = run_pipeline_from_evidence(merged, providers=providers)
    case["errors"] = list(session.chunk_errors) + case.get("errors", [])
    case["call_id"] = call_id
    case["chunks_processed"] = len(session.chunk_bundles)
    return case


def end_voice_session(call_id: str, *, store: InMemoryVoiceSessionStore | None = None) -> None:
    """Call when Layer 0 signals the call ended, to free session memory."""
    (store or _default_store).clear(call_id)
