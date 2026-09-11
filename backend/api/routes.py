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

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_case_store, get_intake_service, get_voice_session_store
from backend.api.schemas import (
    ChatMessageIn,
    IntakeAck,
    PortalSubmitIn,
    VoiceAudioChunkIn,
    VoiceIncomingIn,
    VoiceSessionAck,
)
from backend.ingestion.case_store import InMemoryCaseStore
from backend.ingestion.input_envelope import ConsentStatus, EnvelopeMetadata
from backend.ingestion.intake_service import IntakeService
from backend.ingestion.validator import IntakeValidationError, RawIntakeRequest
from backend.ingestion.voice_session import InMemoryVoiceSessionStore
from backend.models.enums import Channel, Modality

router = APIRouter()


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
    intake_service: IntakeService = Depends(get_intake_service),
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
    return _ack(outcome)


@router.post("/portal/submit", response_model=IntakeAck)
def portal_submit(
    payload: PortalSubmitIn,
    intake_service: IntakeService = Depends(get_intake_service),
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
    return _ack(outcome)


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
    intake_service: IntakeService = Depends(get_intake_service),
    voice_sessions: InMemoryVoiceSessionStore = Depends(get_voice_session_store),
) -> IntakeAck:
    """Accepts one buffered audio chunk reference for an already-started call."""
    modalities = [Modality.AUDIO.value]
    if payload.transcript_text:
        modalities.append(Modality.TEXT.value)

    audio_dict = payload.audio.model_dump()
    audio_dict.setdefault("call_id", payload.call_id)

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

    try:
        voice_sessions.buffer_chunk(payload.call_id, outcome.envelope.audio)
    except KeyError as exc:
        raise IntakeValidationError("UNKNOWN_CALL_SESSION", str(exc), field_name="call_id")

    return _ack(outcome)
