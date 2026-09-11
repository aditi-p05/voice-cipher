"""
Layer 0 — Common Input Envelope.

This is the single normalized shape that ALL three channels (voice call,
chatbot, complaint portal) converge into before being handed off to
Layer 1. Layer 1+ must be able to depend on this shape without caring
which channel produced it.

No PII analysis, no redaction, no risk/vulnerability logic lives here.
This module only defines *structure* and light, deterministic shape
validation (via pydantic field types) — semantic validation lives in
validator.py.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.models.enums import Channel, Modality

ENVELOPE_SCHEMA_VERSION = "1.0.0"


def new_case_id() -> str:
    return f"CASE-{uuid.uuid4().hex[:8].upper()}"


def new_input_id() -> str:
    return f"IN-{uuid.uuid4().hex[:10].upper()}"


class TextContent(BaseModel):
    """Raw textual evidence (chat message, portal free-text, ASR-pending voice note)."""

    body: str = Field(..., min_length=1, max_length=8000)


class AudioReference(BaseModel):
    """
    A reference to audio data/chunks, NOT the audio bytes themselves and NOT
    a transcript. STT / voice-feature extraction is explicitly out of scope
    for Layer 0.
    """

    audio_ref_id: str
    call_id: Optional[str] = None
    mime_type: Optional[str] = None
    duration_seconds: Optional[float] = Field(default=None, ge=0)
    sample_rate_hz: Optional[int] = Field(default=None, gt=0)
    is_chunk: bool = False
    sequence_number: Optional[int] = Field(default=None, ge=0)
    storage_uri: Optional[str] = None  # e.g. path/key in blob storage or buffer handle


class StructuredComplaintData(BaseModel):
    """Structured fields submitted via the complaint portal form."""

    category: Optional[str] = Field(default=None, max_length=200)
    location: Optional[str] = Field(default=None, max_length=300)
    fields: dict[str, str] = Field(default_factory=dict, max_length=50)


class ConsentStatus(BaseModel):
    """
    Minimal consent/status placeholder required by the wider project
    architecture. Layer 0 only records what was declared at intake time;
    it does not evaluate, enforce, or verify consent.
    """

    consent_given: Optional[bool] = None
    consent_source: Optional[str] = None  # e.g. "ivr_prompt", "portal_checkbox"


class EnvelopeMetadata(BaseModel):
    """Free-form, non-sensitive operational metadata. No PII by contract."""

    source_ip_hash: Optional[str] = None
    user_agent: Optional[str] = None
    channel_session_id: Optional[str] = None
    extra: dict[str, str] = Field(default_factory=dict, max_length=20)


class InputEnvelope(BaseModel):
    """
    The common Case/Input Envelope produced by Layer 0 for every accepted
    intake, regardless of channel.
    """

    schema_version: str = ENVELOPE_SCHEMA_VERSION

    input_id: str = Field(default_factory=new_input_id)
    case_id: str

    channel: Channel
    modalities: list[Modality] = Field(..., min_length=1)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language_hint: Optional[str] = None

    text: Optional[TextContent] = None
    audio: Optional[AudioReference] = None
    structured_data: Optional[StructuredComplaintData] = None

    consent: ConsentStatus = Field(default_factory=ConsentStatus)
    metadata: EnvelopeMetadata = Field(default_factory=EnvelopeMetadata)

    model_config = ConfigDict(use_enum_values=False)
