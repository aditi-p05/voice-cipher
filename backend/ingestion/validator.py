"""
Layer 0 — deterministic validation.

No LLMs, no ML, no business/risk rules. Pure structural/deterministic
checks so behavior is predictable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.models.enums import CHANNEL_ALLOWED_MODALITIES, Channel, Modality
from backend.ingestion.input_envelope import (
    AudioReference,
    StructuredComplaintData,
    TextContent,
)

MAX_TEXT_LEN = 8000
MAX_AUDIO_DURATION_SECONDS = 3600  # 1 hour hard ceiling for a single chunk/segment
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/webm",
    "audio/L16",
}


class IntakeValidationError(Exception):
    """Raised for any deterministic Layer 0 validation failure."""

    def __init__(self, code: str, message: str, field_name: str | None = None):
        self.code = code
        self.message = message
        self.field_name = field_name
        super().__init__(message)


@dataclass
class RawIntakeRequest:
    """
    Loosely-typed shape of an inbound request, before it becomes a strict
    InputEnvelope. Kept separate so the API layer can build this from any
    channel's payload uniformly before validation.
    """

    channel: str
    modalities: list[str]
    case_id: str | None = None
    language_hint: str | None = None
    text_body: str | None = None
    audio: dict | None = None
    structured_fields: dict | None = None
    metadata: dict = field(default_factory=dict)


def _validate_channel(channel_raw: str) -> Channel:
    try:
        return Channel(channel_raw)
    except ValueError:
        raise IntakeValidationError(
            "UNSUPPORTED_CHANNEL",
            f"Channel '{channel_raw}' is not supported.",
            field_name="channel",
        )


def _validate_modalities(modalities_raw: list[str], channel: Channel) -> list[Modality]:
    if not modalities_raw:
        raise IntakeValidationError(
            "MISSING_MODALITIES",
            "At least one modality must be declared.",
            field_name="modalities",
        )

    parsed: list[Modality] = []
    for m in modalities_raw:
        try:
            parsed.append(Modality(m))
        except ValueError:
            raise IntakeValidationError(
                "UNSUPPORTED_MODALITY",
                f"Modality '{m}' is not supported.",
                field_name="modalities",
            )

    allowed = CHANNEL_ALLOWED_MODALITIES[channel]
    invalid = [m for m in parsed if m not in allowed]
    if invalid:
        raise IntakeValidationError(
            "INVALID_CHANNEL_MODALITY_COMBINATION",
            f"Modalities {[m.value for m in invalid]} are not valid for channel "
            f"'{channel.value}'.",
            field_name="modalities",
        )

    return parsed


def _validate_case_id(case_id: str | None) -> None:
    if case_id is None:
        return
    if not case_id.strip():
        raise IntakeValidationError(
            "INVALID_CASE_ID", "case_id, if provided, must be non-empty.", field_name="case_id"
        )
    if not case_id.startswith("CASE-"):
        raise IntakeValidationError(
            "INVALID_CASE_ID",
            "case_id must follow the 'CASE-XXXX' format.",
            field_name="case_id",
        )


def _validate_text(text_body: str | None, modalities: list[Modality]) -> TextContent | None:
    if Modality.TEXT not in modalities:
        return None
    if not text_body or not text_body.strip():
        raise IntakeValidationError(
            "MISSING_TEXT_CONTENT",
            "Modality 'text' declared but no text content was provided.",
            field_name="text",
        )
    if len(text_body) > MAX_TEXT_LEN:
        raise IntakeValidationError(
            "TEXT_TOO_LONG",
            f"Text content exceeds max length of {MAX_TEXT_LEN} characters.",
            field_name="text",
        )
    return TextContent(body=text_body)


def _validate_audio(
    audio_raw: dict | None, modalities: list[Modality]
) -> AudioReference | None:
    if Modality.AUDIO not in modalities:
        return None
    if not audio_raw:
        raise IntakeValidationError(
            "MISSING_AUDIO_METADATA",
            "Modality 'audio' declared but no audio metadata was provided.",
            field_name="audio",
        )

    mime_type = audio_raw.get("mime_type")
    if mime_type and mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
        raise IntakeValidationError(
            "UNSUPPORTED_AUDIO_FORMAT",
            f"Audio mime type '{mime_type}' is not supported.",
            field_name="audio.mime_type",
        )

    duration = audio_raw.get("duration_seconds")
    if duration is not None and (duration < 0 or duration > MAX_AUDIO_DURATION_SECONDS):
        raise IntakeValidationError(
            "INVALID_AUDIO_DURATION",
            "Audio duration is out of the accepted range.",
            field_name="audio.duration_seconds",
        )

    if not audio_raw.get("audio_ref_id"):
        raise IntakeValidationError(
            "MISSING_AUDIO_REF_ID",
            "audio.audio_ref_id is required when modality 'audio' is declared.",
            field_name="audio.audio_ref_id",
        )

    try:
        return AudioReference(
            audio_ref_id=audio_raw.get("audio_ref_id"),
            call_id=audio_raw.get("call_id"),
            mime_type=mime_type,
            duration_seconds=duration,
            sample_rate_hz=audio_raw.get("sample_rate_hz"),
            is_chunk=bool(audio_raw.get("is_chunk", False)),
            sequence_number=audio_raw.get("sequence_number"),
            storage_uri=audio_raw.get("storage_uri"),
        )
    except Exception as exc:  # pydantic ValidationError -> deterministic 4xx
        raise IntakeValidationError(
            "MALFORMED_AUDIO_METADATA", f"Audio metadata malformed: {exc}", field_name="audio"
        )


def _validate_structured(
    structured_raw: dict | None, modalities: list[Modality]
) -> StructuredComplaintData | None:
    if Modality.STRUCTURED_DATA not in modalities:
        return None
    if not structured_raw:
        raise IntakeValidationError(
            "MISSING_STRUCTURED_DATA",
            "Modality 'structured_data' declared but no structured fields were provided.",
            field_name="structured_data",
        )
    try:
        return StructuredComplaintData(
            category=structured_raw.get("category"),
            location=structured_raw.get("location"),
            fields=structured_raw.get("fields", {}) or {},
        )
    except Exception as exc:
        raise IntakeValidationError(
            "MALFORMED_STRUCTURED_DATA",
            f"Structured data malformed: {exc}",
            field_name="structured_data",
        )


@dataclass
class ValidatedIntake:
    channel: Channel
    modalities: list[Modality]
    case_id: str | None
    text: TextContent | None
    audio: AudioReference | None
    structured_data: StructuredComplaintData | None


def validate_intake(raw: RawIntakeRequest) -> ValidatedIntake:
    """
    Run all deterministic Layer 0 checks. Raises IntakeValidationError on the
    first failure encountered.
    """
    channel = _validate_channel(raw.channel)
    modalities = _validate_modalities(raw.modalities, channel)
    _validate_case_id(raw.case_id)

    text = _validate_text(raw.text_body, modalities)
    audio = _validate_audio(raw.audio, modalities)
    structured = _validate_structured(raw.structured_fields, modalities)

    return ValidatedIntake(
        channel=channel,
        modalities=modalities,
        case_id=raw.case_id,
        text=text,
        audio=audio,
        structured_data=structured,
    )
