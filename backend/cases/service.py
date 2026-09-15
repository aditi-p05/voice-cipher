from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from backend.orchestration.audit import record_operator_action
from backend.ingestion.input_envelope import InputEnvelope
from backend.cases.store import InMemoryCaseReadStore

if TYPE_CHECKING:
    from backend.orchestration.voice_session import InMemoryVoiceSessionStore


class CaseNotFoundError(KeyError): pass
class OperatorActionError(ValueError): pass


class CaseIntegrationService:
    def __init__(self, store: InMemoryCaseReadStore) -> None: self._store = store

    def process(self, envelope: InputEnvelope) -> dict[str, Any]:
        # Import lazily: Layer 0 remains usable if optional orchestration
        # dependencies are unavailable in a minimal ingestion deployment.
        try:
            from backend.orchestration.run import run_pipeline
            view = run_pipeline(envelope)
        except Exception as exc:
            view = {
                "schema_version": "1.0.0", "case_id": envelope.case_id,
                "input_id": envelope.input_id, "channel": envelope.channel.value,
                "evidence_available": False, "evidence_bundle": None,
                "svi_result": None, "risk_tier": None,
                "service_recommendation": None, "support_decision": None,
                "requires_operator_confirmation": False,
                "errors": [{"stage": "pipeline", "code": "PIPELINE_UNAVAILABLE", "message": "Pipeline output is temporarily unavailable."}],
            }
        self._store.upsert_pipeline_view(view)
        return view

    def process_voice_chunk(
        self,
        envelope: InputEnvelope,
        *,
        voice_store: InMemoryVoiceSessionStore,
    ) -> dict[str, Any]:
        """Process a voice chunk through Layer 4B's cumulative session state."""
        # Keep the orchestration dependency lazy so a minimal Layer 0
        # deployment continues to degrade safely when it is unavailable.
        try:
            from backend.orchestration.voice_session import ingest_voice_chunk

            view = ingest_voice_chunk(envelope, store=voice_store)
        except Exception as exc:
            view = {
                "schema_version": "1.0.0", "case_id": envelope.case_id,
                "input_id": envelope.input_id, "channel": envelope.channel.value,
                "evidence_available": False, "evidence_bundle": None,
                "svi_result": None, "risk_tier": None,
                "service_recommendation": None, "support_decision": None,
                "requires_operator_confirmation": False,
                "errors": [{"stage": "pipeline", "code": "PIPELINE_UNAVAILABLE", "message": "Pipeline output is temporarily unavailable."}],
            }
        self._store.upsert_pipeline_view(view)
        return view

    def list_cases(self) -> list[dict[str, Any]]:
        return [self._summary(v) for v in self._store.list()]

    def detail(self, case_id: str) -> dict[str, Any]:
        view = self._store.get(case_id)
        if view is None: raise CaseNotFoundError(case_id)
        view["operator_actions"] = self._store.actions(case_id)
        view["operator_confirmation_state"] = view["operator_actions"][-1]["action"] if view["operator_actions"] else "PENDING"
        return view

    def act(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        view = self._store.get(case_id)
        if view is None: raise CaseNotFoundError(case_id)
        action = payload["action"]
        if action not in {"CONFIRM", "OVERRIDE"}: raise OperatorActionError("action must be CONFIRM or OVERRIDE")
        if not payload["operator_id"].strip(): raise OperatorActionError("operator_id is required")
        if action == "OVERRIDE" and not (payload.get("reason") or "").strip(): raise OperatorActionError("reason is required for OVERRIDE")
        recommendation = view.get("service_recommendation")
        record = {"schema_version": "1.0.0", "case_id": case_id, "operator_id": payload["operator_id"], "action": action, "timestamp": datetime.now(timezone.utc).isoformat(), "original_recommendation": recommendation, "final_decision": payload.get("final_decision"), "reason": payload.get("reason")}
        record_operator_action(record)
        self._store.append_action(record)
        return record

    @staticmethod
    def _summary(v: dict[str, Any]) -> dict[str, Any]:
        e, s = v.get("evidence_bundle") or {}, v.get("svi_result") or {}
        return {"case_id": v["case_id"], "channel": v.get("channel"), "language": (e.get("transcript") or {}).get("language"), "timestamp": v.get("updated_at"), "status": "NEEDS_OPERATOR_REVIEW" if v.get("requires_operator_confirmation") else "PROCESSING", "risk_tier": v.get("risk_tier"), "svi_score": s.get("svi_score"), "operator_confirmation_state": "PENDING"}
