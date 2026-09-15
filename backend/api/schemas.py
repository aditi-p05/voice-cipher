"""
Layer 0 — API-facing request/response schemas.

Kept separate from the internal InputEnvelope so the wire format can evolve
independently of the internal model.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.ingestion.input_envelope import ENVELOPE_SCHEMA_VERSION


class AudioMetadataIn(BaseModel):
    audio_ref_id: str
    call_id: Optional[str] = None
    mime_type: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate_hz: Optional[int] = None
    is_chunk: bool = False
    sequence_number: Optional[int] = None
    storage_uri: Optional[str] = None


class StructuredDataIn(BaseModel):
    category: Optional[str] = None
    location: Optional[str] = None
    fields: dict[str, str] = Field(default_factory=dict)


class ChatMessageIn(BaseModel):
    case_id: Optional[str] = None
    message: str
    language_hint: Optional[str] = None
    channel_session_id: Optional[str] = None


class PortalSubmitIn(BaseModel):
    case_id: Optional[str] = None
    complaint_text: Optional[str] = None
    structured_data: Optional[StructuredDataIn] = None
    audio: Optional[AudioMetadataIn] = None
    language_hint: Optional[str] = None
    consent_given: Optional[bool] = None


class VoiceIncomingIn(BaseModel):
    case_id: Optional[str] = None
    call_id: str
    language_hint: Optional[str] = None
    consent_given: Optional[bool] = None


class VoiceSessionAck(BaseModel):
    """Acknowledgement for a voice call-start event, before evidence arrives."""

    schema_version: str = ENVELOPE_SCHEMA_VERSION
    case_id: str
    call_id: str
    accepted: bool


class VoiceAudioChunkIn(BaseModel):
    case_id: str
    call_id: str
    audio: AudioMetadataIn
    transcript_text: Optional[str] = None  # only if the provider streams interim text alongside


class VoiceSessionEndedIn(BaseModel):
    call_id: str


class IntakeAck(BaseModel):
    schema_version: str = ENVELOPE_SCHEMA_VERSION
    case_id: str
    input_id: str
    channel: str
    modalities: list[str]
    accepted: bool
    dispatch_queue: str

class OperatorActionIn(BaseModel):
    operator_id: str = Field(min_length=1, max_length=200)
    action: str
    reason: Optional[str] = Field(default=None, max_length=2000)
    final_decision: Optional[str] = Field(default=None, max_length=4000)
