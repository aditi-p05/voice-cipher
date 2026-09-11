"""
Layer 0 — intake orchestration.

Wires together: validation -> case association -> envelope construction ->
modality routing -> dispatch -> structured logging.

This is the single place channel-specific API routes should call into, so
chat/portal/voice all go through identical, testable logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.core.logging import log_event
from backend.ingestion.case_store import CaseStore
from backend.ingestion.input_envelope import (
    ConsentStatus,
    EnvelopeMetadata,
    InputEnvelope,
)
from backend.ingestion.validator import RawIntakeRequest, ValidatedIntake, validate_intake
from backend.routing.modality_router import RoutingResult, route
from backend.transport.dispatcher import DispatchResult, InputDispatcher


@dataclass
class IntakeOutcome:
    envelope: InputEnvelope
    routing: RoutingResult
    dispatch: DispatchResult


class IntakeService:
    def __init__(self, case_store: CaseStore, dispatcher: InputDispatcher):
        self._case_store = case_store
        self._dispatcher = dispatcher

    def _resolve_case_id(self, requested_case_id: str | None) -> str:
        if requested_case_id and self._case_store.exists(requested_case_id):
            return requested_case_id
        if requested_case_id and not self._case_store.exists(requested_case_id):
            # Caller referenced a case_id we don't know about yet.
            # Treat as authoritative and register it, rather than silently
            # minting a different id (avoids orphaning a client's records).
            self._case_store.register_case_id(requested_case_id)
            return requested_case_id
        return self._case_store.create_case()

    def handle_intake(
        self,
        raw: RawIntakeRequest,
        language_hint: str | None = None,
        consent: ConsentStatus | None = None,
        metadata: EnvelopeMetadata | None = None,
    ) -> IntakeOutcome:
        validated: ValidatedIntake = validate_intake(raw)

        case_id = self._resolve_case_id(validated.case_id)

        envelope = InputEnvelope(
            case_id=case_id,
            channel=validated.channel,
            modalities=validated.modalities,
            language_hint=language_hint,
            text=validated.text,
            audio=validated.audio,
            structured_data=validated.structured_data,
            consent=consent or ConsentStatus(),
            metadata=metadata or EnvelopeMetadata(),
        )

        self._case_store.register_input(case_id, envelope.input_id)

        routing_result = route(envelope)

        log_event(
            "input_received",
            case_id=envelope.case_id,
            input_id=envelope.input_id,
            channel=envelope.channel.value,
            modalities=[m.value for m in envelope.modalities],
        )
        log_event(
            "validation_result",
            case_id=envelope.case_id,
            input_id=envelope.input_id,
            result="passed",
        )

        dispatch_result = self._dispatcher.publish(envelope)

        log_event(
            "dispatch_result",
            case_id=envelope.case_id,
            input_id=envelope.input_id,
            accepted=dispatch_result.accepted,
            queue=dispatch_result.queue_name,
        )

        return IntakeOutcome(envelope=envelope, routing=routing_result, dispatch=dispatch_result)
