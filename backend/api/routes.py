"""
Layer 0 — API routes.

    /chat/message     chatbot ingestion
    /portal/submit    complaint portal ingestion
    /voice/incoming   voice call/session identification
    /voice/chunk      buffered audio chunk ingestion for an existing call

Each route only: validates, builds an Input Envelope, routes modalities,
and dispatches downstream. No NLP/LLM/ML/risk logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.dependencies import (
    get_case_integration_service,
    get_case_store,
    get_intake_service,
    get_orchestration_voice_sessions,
    get_voice_session_store,
)
from backend.api.schemas import (
    ChatMessageIn,
    IntakeAck,
    PortalSubmitIn,
    VoiceAudioChunkIn,
    VoiceIncomingIn,
    VoiceSessionEndedIn,
    VoiceSessionAck,
    OperatorActionIn,
)
from backend.ingestion.case_store import InMemoryCaseStore
from backend.ingestion.input_envelope import ConsentStatus, EnvelopeMetadata
from backend.ingestion.intake_service import IntakeService
from backend.ingestion.validator import IntakeValidationError, RawIntakeRequest
from backend.ingestion.voice_session import InMemoryVoiceSessionStore
from backend.orchestration.voice_session import (
    InMemoryVoiceSessionStore as OrchestrationVoiceSessionStore,
    end_voice_session,
)
from backend.models.enums import Channel, Modality
from backend.cases.service import CaseIntegrationService, CaseNotFoundError, OperatorActionError

router = APIRouter()

def _integrate(outcome, service: CaseIntegrationService) -> None:
    # Pipeline errors are represented in the case view; a successful intake is never undone.
    service.process(outcome.envelope)


def _ack(outcome) -> IntakeAck:
    return IntakeAck(
        case_id=outcome.envelope.case_id,
        input_id=outcome.envelope.input_id,
        channel=outcome.envelope.channel.value,
        modalities=[m.value for m in outcome.envelope.modalities],
        accepted=outcome.dispatch.accepted,
        dispatch_queue=outcome.dispatch.queue_name,
    )


@router.post("/chat/message", response_model=IntakeAck)
def chat_message(
    payload: ChatMessageIn,
    intake_service: IntakeService = Depends(get_intake_service), integration: CaseIntegrationService = Depends(get_case_integration_service),
) -> IntakeAck:
    raw = RawIntakeRequest(
        channel=Channel.CHATBOT.value,
        modalities=[Modality.TEXT.value],
        case_id=payload.case_id,
        text_body=payload.message,
    )
    outcome = intake_service.handle_intake(
        raw,
        language_hint=payload.language_hint,
        metadata=EnvelopeMetadata(channel_session_id=payload.channel_session_id),
    )
    _integrate(outcome, integration); return _ack(outcome)


@router.post("/portal/submit", response_model=IntakeAck)
def portal_submit(
    payload: PortalSubmitIn,
    intake_service: IntakeService = Depends(get_intake_service), integration: CaseIntegrationService = Depends(get_case_integration_service),
) -> IntakeAck:
    modalities: list[str] = []
    if payload.complaint_text:
        modalities.append(Modality.TEXT.value)
    if payload.structured_data:
        modalities.append(Modality.STRUCTURED_DATA.value)
    if payload.audio:
        modalities.append(Modality.AUDIO.value)

    raw = RawIntakeRequest(
        channel=Channel.PORTAL.value,
        modalities=modalities,
        case_id=payload.case_id,
        text_body=payload.complaint_text,
        structured_fields=(
            payload.structured_data.model_dump() if payload.structured_data else None
        ),
        audio=payload.audio.model_dump() if payload.audio else None,
    )
    outcome = intake_service.handle_intake(
        raw,
        language_hint=payload.language_hint,
        consent=ConsentStatus(consent_given=payload.consent_given, consent_source="portal_form"),
    )
    _integrate(outcome, integration); return _ack(outcome)


@router.post("/voice/incoming", response_model=VoiceSessionAck)
def voice_incoming(
    payload: VoiceIncomingIn,
    case_store: InMemoryCaseStore = Depends(get_case_store),
    voice_sessions: InMemoryVoiceSessionStore = Depends(get_voice_session_store),
) -> VoiceSessionAck:
    """
    Called by the Twilio/Exotel webhook when a call starts. Establishes the
    call<->case association; no evidence or InputEnvelope exists at this point.
    Audio evidence is accepted later through /voice/chunk.
    """
    if payload.case_id:
        if not case_store.exists(payload.case_id):
            case_store.register_case_id(payload.case_id)
        case_id = payload.case_id
    else:
        case_id = case_store.create_case()

    voice_sessions.start_session(call_id=payload.call_id, case_id=case_id)
    return VoiceSessionAck(case_id=case_id, call_id=payload.call_id, accepted=True)


@router.post("/voice/chunk", response_model=IntakeAck)
def voice_audio_chunk(
    payload: VoiceAudioChunkIn,
    intake_service: IntakeService = Depends(get_intake_service), integration: CaseIntegrationService = Depends(get_case_integration_service),
    voice_sessions: InMemoryVoiceSessionStore = Depends(get_voice_session_store),
    orchestration_voice_sessions: OrchestrationVoiceSessionStore = Depends(get_orchestration_voice_sessions),
) -> IntakeAck:
    """Accept one audio chunk for an already-started call."""
    modalities = [Modality.AUDIO.value]
    if payload.transcript_text:
        modalities.append(Modality.TEXT.value)

    audio_dict = payload.audio.model_dump()
    audio_dict["call_id"] = payload.call_id

    raw = RawIntakeRequest(
        channel=Channel.VOICE_CALL.value,
        modalities=modalities,
        case_id=payload.case_id,
        text_body=payload.transcript_text,
        audio=audio_dict,
    )
    outcome = intake_service.handle_intake(
        raw, metadata=EnvelopeMetadata(channel_session_id=payload.call_id)
    )

    if voice_sessions.get_session(payload.call_id) is None:
        raise IntakeValidationError(
            "UNKNOWN_CALL_SESSION",
            str(KeyError(
                f"Unknown call_id '{payload.call_id}'; call /voice/incoming first."
            )),
            field_name="call_id",
        )

    integration.process_voice_chunk(
        outcome.envelope,
        voice_store=orchestration_voice_sessions,
    )
    return _ack(outcome)


@router.post("/voice/ended")
def voice_ended(
    payload: VoiceSessionEndedIn,
    voice_sessions: InMemoryVoiceSessionStore = Depends(get_voice_session_store),
    orchestration_voice_sessions: OrchestrationVoiceSessionStore = Depends(get_orchestration_voice_sessions),
) -> dict[str, str | bool]:
    """End a call and clear both its identity and cumulative evidence state."""
    end_voice_session(payload.call_id, store=orchestration_voice_sessions)
    voice_sessions.end_session(payload.call_id)
    return {"call_id": payload.call_id, "ended": True}

@router.get("/cases")
def list_cases(service: CaseIntegrationService = Depends(get_case_integration_service)) -> dict:
    return {"cases": service.list_cases()}

@router.get("/case/{case_id}")
def get_case(case_id: str, service: CaseIntegrationService = Depends(get_case_integration_service)) -> dict:
    try: return service.detail(case_id)
    except CaseNotFoundError: raise HTTPException(404, detail={"code": "CASE_NOT_FOUND", "message": "Case was not found."})

@router.post("/case/{case_id}/confirm")
def confirm_case(case_id: str, payload: OperatorActionIn, service: CaseIntegrationService = Depends(get_case_integration_service)) -> dict:
    try: return service.act(case_id, payload.model_dump())
    except CaseNotFoundError: raise HTTPException(404, detail={"code": "CASE_NOT_FOUND", "message": "Case was not found."})
    except OperatorActionError as exc: raise HTTPException(400, detail={"code": "INVALID_OPERATOR_ACTION", "message": str(exc)})
