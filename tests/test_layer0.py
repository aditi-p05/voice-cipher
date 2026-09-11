import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.dependencies import get_case_store, get_dispatcher, get_voice_session_store
from backend.core.logging import log_event, logger
from backend.models.enums import Channel, Modality
from backend.routing.modality_router import route
from backend.ingestion.input_envelope import InputEnvelope

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Chat intake creates a valid Input Envelope
# ---------------------------------------------------------------------------
def test_chat_intake_creates_valid_envelope():
    resp = client.post("/chat/message", json={"message": "I need help with my complaint."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["channel"] == Channel.CHATBOT.value
    assert body["modalities"] == [Modality.TEXT.value]
    assert body["case_id"].startswith("CASE-")
    assert body["accepted"] is True


# ---------------------------------------------------------------------------
# 2. Portal intake creates a valid Input Envelope
# ---------------------------------------------------------------------------
def test_portal_intake_creates_valid_envelope():
    resp = client.post(
        "/portal/submit",
        json={
            "complaint_text": "My complaint is about a road safety issue.",
            "structured_data": {"category": "road_safety", "location": "NH-44"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["channel"] == Channel.PORTAL.value
    assert set(body["modalities"]) == {Modality.TEXT.value, Modality.STRUCTURED_DATA.value}


# ---------------------------------------------------------------------------
# 3. Voice intake accepts audio metadata/chunks
# ---------------------------------------------------------------------------
def test_voice_intake_accepts_audio_chunk():
    start = client.post("/voice/incoming", json={"call_id": "CALL-1"})
    assert start.status_code == 200
    case_id = start.json()["case_id"]

    chunk = client.post(
        "/voice/chunk",
        json={
            "case_id": case_id,
            "call_id": "CALL-1",
            "audio": {
                "audio_ref_id": "AUD-1",
                "mime_type": "audio/wav",
                "duration_seconds": 4.2,
                "is_chunk": True,
                "sequence_number": 0,
            },
        },
    )
    assert chunk.status_code == 200
    body = chunk.json()
    assert Modality.AUDIO.value in body["modalities"]

    voice_sessions = get_voice_session_store()
    session = voice_sessions.get_session("CALL-1")
    assert session is not None
    assert len(session.chunks) == 1
    assert session.chunks[0].audio_ref_id == "AUD-1"


# ---------------------------------------------------------------------------
# 4. Multiple chat inputs can belong to the same case
# ---------------------------------------------------------------------------
def test_multiple_chat_inputs_same_case():
    first = client.post("/chat/message", json={"message": "First message"})
    case_id = first.json()["case_id"]

    second = client.post(
        "/chat/message", json={"case_id": case_id, "message": "Second message, same case"}
    )
    assert second.status_code == 200
    assert second.json()["case_id"] == case_id

    case_store = get_case_store()
    input_ids = case_store.input_ids_for_case(case_id)
    assert len(input_ids) == 2


# ---------------------------------------------------------------------------
# 5. Channel and modality remain separate
# ---------------------------------------------------------------------------
def test_channel_and_modality_are_independent_concepts():
    envelope = InputEnvelope(
        case_id="CASE-SEPTEST1",
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO],
    )
    # Same channel ("portal") can carry very different modality sets.
    envelope_minimal = InputEnvelope(
        case_id="CASE-SEPTEST1",
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT],
    )
    assert envelope.channel == envelope_minimal.channel
    assert envelope.modalities != envelope_minimal.modalities


# ---------------------------------------------------------------------------
# 6. Invalid channel is rejected
# ---------------------------------------------------------------------------
def test_invalid_channel_is_rejected():
    resp = client.post(
        "/portal/submit",
        json={"complaint_text": "hello"},
    )
    assert resp.status_code == 200  # sanity: valid case still works

    # Directly exercise the validator with a bad channel, since all current
    # routes hardcode a valid channel value.
    from backend.ingestion.validator import RawIntakeRequest, validate_intake, IntakeValidationError

    raw = RawIntakeRequest(channel="carrier_pigeon", modalities=["text"], text_body="hi")
    with pytest.raises(IntakeValidationError) as exc_info:
        validate_intake(raw)
    assert exc_info.value.code == "UNSUPPORTED_CHANNEL"


# ---------------------------------------------------------------------------
# 7. Invalid modality is rejected
# ---------------------------------------------------------------------------
def test_invalid_modality_is_rejected():
    from backend.ingestion.validator import RawIntakeRequest, validate_intake, IntakeValidationError

    raw = RawIntakeRequest(channel="chatbot", modalities=["telepathy"], text_body="hi")
    with pytest.raises(IntakeValidationError) as exc_info:
        validate_intake(raw)
    assert exc_info.value.code == "UNSUPPORTED_MODALITY"


def test_invalid_channel_modality_combination_is_rejected():
    from backend.ingestion.validator import RawIntakeRequest, validate_intake, IntakeValidationError

    # chatbot channel does not permit structured_data modality
    raw = RawIntakeRequest(
        channel="chatbot", modalities=["structured_data"], structured_fields={"category": "x"}
    )
    with pytest.raises(IntakeValidationError) as exc_info:
        validate_intake(raw)
    assert exc_info.value.code == "INVALID_CHANNEL_MODALITY_COMBINATION"


# ---------------------------------------------------------------------------
# 8. Missing required fields are rejected
# ---------------------------------------------------------------------------
def test_missing_text_content_is_rejected():
    resp = client.post("/chat/message", json={"message": ""})
    assert resp.status_code in (400, 422)


def test_missing_structured_data_is_rejected_at_validator_level():
    from backend.ingestion.validator import RawIntakeRequest, validate_intake, IntakeValidationError

    raw = RawIntakeRequest(channel="portal", modalities=["structured_data"])
    with pytest.raises(IntakeValidationError) as exc_info:
        validate_intake(raw)
    assert exc_info.value.code == "MISSING_STRUCTURED_DATA"


def test_missing_audio_ref_id_is_rejected():
    from backend.ingestion.validator import RawIntakeRequest, validate_intake, IntakeValidationError

    raw = RawIntakeRequest(
        channel="voice_call",
        modalities=["audio"],
        audio={"mime_type": "audio/wav"},  # no audio_ref_id
    )
    with pytest.raises(IntakeValidationError) as exc_info:
        validate_intake(raw)
    assert exc_info.value.code == "MISSING_AUDIO_REF_ID"


# ---------------------------------------------------------------------------
# 9. Router correctly identifies available modalities
# ---------------------------------------------------------------------------
def test_router_identifies_modalities_for_portal_with_audio():
    envelope = InputEnvelope(
        case_id="CASE-ROUTETST",
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO],
    )
    result = route(envelope)
    assert result.has_text is True
    assert result.has_structured_data is True
    assert result.has_audio is True
    assert result.transcript_expected is True


def test_router_never_returns_risk_fields():
    envelope = InputEnvelope(
        case_id="CASE-ROUTETST2", channel=Channel.CHATBOT, modalities=[Modality.TEXT]
    )
    result = route(envelope)
    result_fields = set(vars(result).keys())
    forbidden = {"risk_tier", "vulnerability_score", "svi", "escalate", "is_critical"}
    assert result_fields.isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# 10. Dispatcher receives the validated envelope
# ---------------------------------------------------------------------------
def test_dispatcher_receives_envelope():
    dispatcher = get_dispatcher()
    before = len(dispatcher.published)

    resp = client.post("/chat/message", json={"message": "Dispatch check message"})
    assert resp.status_code == 200

    after = len(dispatcher.published)
    assert after == before + 1
    assert dispatcher.published[-1].text.body == "Dispatch check message"


# ---------------------------------------------------------------------------
# 11. Raw sensitive content is not written to normal logs
# ---------------------------------------------------------------------------
def test_sensitive_content_not_logged(caplog):
    caplog.set_level(logging.INFO, logger="nhaa.layer0")
    secret_text = "SUPER_SECRET_COMPLAINT_DETAIL_98765"

    resp = client.post("/chat/message", json={"message": secret_text})
    assert resp.status_code == 200

    for record in caplog.records:
        assert secret_text not in record.message


def test_log_event_rejects_forbidden_fields():
    with pytest.raises(ValueError):
        log_event("input_received", text="this should never be allowed")
