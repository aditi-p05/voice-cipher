"""
Layer 1 (Fusion / Privacy) — Member 2 test suite.

All inputs are synthetic/fabricated. No real complainant data is used
anywhere in this file, per the Layer 1 task brief and CONTRACTS.md's
data-safety expectations.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.ingestion.input_envelope import (
    AudioReference,
    InputEnvelope,
    StructuredComplaintData,
    TextContent,
)
from backend.models.enums import Channel, Modality
from services.fusion import EvidenceBundle, FusionError, build_evidence_bundle
from services.fusion.stt.base import STTResult, STTSegment
from services.fusion.stt.mock_provider import MockSTTProvider
from services.fusion.voice_features.extractor import FixtureVoiceFeatureExtractor


def _envelope(**overrides) -> InputEnvelope:
    """Small helper for building synthetic InputEnvelopes for these tests."""
    defaults = dict(
        case_id="CASE-TEST0001",
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        text=TextContent(body="This is a synthetic test message."),
    )
    defaults.update(overrides)
    return InputEnvelope(**defaults)


# ---------------------------------------------------------------------------
# 1. Chatbot text
# ---------------------------------------------------------------------------
def test_chatbot_text_produces_redacted_transcript():
    envelope = _envelope(
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        text=TextContent(body="I am scared, he threatened to hurt me."),
    )
    bundle = build_evidence_bundle(envelope)

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.case_id == envelope.case_id
    assert bundle.source_input_ids == [envelope.input_id]
    assert bundle.transcript is not None
    assert bundle.pii_redacted is True
    marker_types = {m.type for m in bundle.markers}
    assert "fear" in marker_types
    assert "threat" in marker_types


# ---------------------------------------------------------------------------
# 2. Portal free text
# ---------------------------------------------------------------------------
def test_portal_text_is_processed():
    envelope = _envelope(
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT],
        text=TextContent(body="My name is Priya Sharma and my phone is 9876543210."),
    )
    bundle = build_evidence_bundle(envelope)

    assert bundle.transcript is not None
    assert "9876543210" not in bundle.transcript.text
    assert "Priya Sharma" not in bundle.transcript.text
    assert bundle.pii_redacted is True


# ---------------------------------------------------------------------------
# 3. Structured portal input
# ---------------------------------------------------------------------------
def test_structured_portal_input_is_processed():
    envelope = _envelope(
        channel=Channel.PORTAL,
        modalities=[Modality.STRUCTURED_DATA],
        text=None,
        structured_data=StructuredComplaintData(
            category="harassment",
            location="test_area",
            fields={"contact_email": "synthetic.user@example.com"},
        ),
    )
    bundle = build_evidence_bundle(envelope)

    assert bundle.transcript is not None
    assert "synthetic.user@example.com" not in bundle.transcript.text
    assert "harassment" in bundle.transcript.text  # non-PII category text preserved


# ---------------------------------------------------------------------------
# 4. Voice with a (synthetic, fixture-registered) transcript
# ---------------------------------------------------------------------------
def test_voice_with_transcript_via_stt_fixture():
    envelope = _envelope(
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        text=None,
        audio=AudioReference(audio_ref_id="AUD-SYN-001"),
    )
    provider = MockSTTProvider(
        fixtures={
            "AUD-SYN-001": STTResult(
                text="He hit me and I am terrified.",
                language="en",
                confidence=0.92,
                segments=[STTSegment(text="He hit me and I am terrified.", start_seconds=0.0, end_seconds=2.5)],
            )
        }
    )

    bundle = build_evidence_bundle(envelope, stt_provider=provider)

    assert bundle.transcript is not None
    assert bundle.transcript.confidence == pytest.approx(0.92)
    assert bundle.transcript.language == "en"
    assert len(bundle.transcript.segments) == 1
    marker_types = {m.type for m in bundle.markers}
    assert "violence" in marker_types
    assert "fear" in marker_types


# ---------------------------------------------------------------------------
# 5. Voice with only an audio reference (no real/fixture transcript)
# ---------------------------------------------------------------------------
def test_voice_with_audio_reference_only_has_no_fabricated_transcript():
    envelope = _envelope(
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        text=None,
        audio=AudioReference(audio_ref_id="AUD-UNKNOWN-999"),
    )
    bundle = build_evidence_bundle(envelope)  # default MockSTTProvider, no fixture

    # No real audio bytes ever reach Layer 1, so nothing should be fabricated.
    assert bundle.transcript is None
    assert bundle.markers == []
    assert bundle.sentiment is None


# ---------------------------------------------------------------------------
# 6. Missing audio (text-only envelope; audio field absent)
# ---------------------------------------------------------------------------
def test_missing_audio_field_does_not_break_pipeline():
    envelope = _envelope(
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        audio=None,
        text=TextContent(body="No audio was ever provided for this case."),
    )
    bundle = build_evidence_bundle(envelope)

    assert bundle.voice_features is None
    assert bundle.transcript is not None


# ---------------------------------------------------------------------------
# 7. Unsupported / unknown language handled gracefully
# ---------------------------------------------------------------------------
def test_unknown_language_is_handled_gracefully():
    envelope = _envelope(
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        language_hint="xx-not-a-real-language",
        text=TextContent(body="."),  # too short/ambiguous for real detection
    )
    # Must not raise; must gracefully fall back rather than crash.
    bundle = build_evidence_bundle(envelope)
    assert isinstance(bundle, EvidenceBundle)


# ---------------------------------------------------------------------------
# 8. PII redaction
# ---------------------------------------------------------------------------
def test_pii_redaction_covers_multiple_entity_types():
    envelope = _envelope(
        text=TextContent(
            body=(
                "This is Mr. Rohan Verma. Email me at rohan.verma@example.com "
                "or call 9123456780. My Aadhaar is 1234 5678 9012."
            )
        )
    )
    bundle = build_evidence_bundle(envelope)
    redacted = bundle.transcript.text

    assert "rohan.verma@example.com" not in redacted
    assert "9123456780" not in redacted
    assert "1234 5678 9012" not in redacted
    assert "Rohan Verma" not in redacted
    assert "REDACTED" in redacted


# ---------------------------------------------------------------------------
# 9. Marker extraction
# ---------------------------------------------------------------------------
def test_marker_extraction_is_explainable():
    envelope = _envelope(text=TextContent(body="He said he would kill me if I report this."))
    bundle = build_evidence_bundle(envelope)

    assert len(bundle.markers) > 0
    for marker in bundle.markers:
        assert 0.0 <= marker.confidence <= 1.0
        assert marker.value  # cue phrase must be present/explainable
        assert marker.type in {
            "threat", "fear", "retaliation", "unsafe", "coercion",
            "violence", "self_harm", "weapon", "immediate_danger",
        }


# ---------------------------------------------------------------------------
# 10. Sentiment
# ---------------------------------------------------------------------------
def test_sentiment_valence_and_arousal_within_range():
    envelope = _envelope(text=TextContent(body="I am scared and shaking, please help me right now!"))
    bundle = build_evidence_bundle(envelope)

    assert bundle.sentiment is not None
    assert -1.0 <= bundle.sentiment.valence <= 1.0
    assert 0.0 <= bundle.sentiment.arousal <= 1.0
    assert bundle.sentiment.valence < 0  # clearly negative synthetic text


# ---------------------------------------------------------------------------
# 11. Missing voice features (must not break the pipeline)
# ---------------------------------------------------------------------------
def test_missing_voice_features_defaults_to_none():
    envelope = _envelope(
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO, Modality.TEXT],
        text=TextContent(body="synthetic interim transcript text"),
        audio=AudioReference(audio_ref_id="AUD-SYN-002"),
    )
    bundle = build_evidence_bundle(envelope)  # default NullVoiceFeatureExtractor
    assert bundle.voice_features is None
    assert isinstance(bundle, EvidenceBundle)  # pipeline did not break


def test_voice_features_present_when_fixture_registered():
    envelope = _envelope(
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        text=None,
        audio=AudioReference(audio_ref_id="AUD-SYN-003"),
    )
    extractor = FixtureVoiceFeatureExtractor(
        fixtures={"AUD-SYN-003": {"pitch_hz_mean": 210.5, "jitter": 0.02}}
    )
    bundle = build_evidence_bundle(envelope, voice_feature_extractor=extractor)

    assert bundle.voice_features is not None
    assert bundle.voice_features.pitch_hz_mean == pytest.approx(210.5)


# ---------------------------------------------------------------------------
# 12. Mixed modalities (portal with audio + text + structured data)
# ---------------------------------------------------------------------------
def test_mixed_modalities_portal_with_audio_and_text_and_structured_data():
    envelope = _envelope(
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO],
        text=TextContent(body="Synthetic portal complaint text."),
        structured_data=StructuredComplaintData(category="test_category"),
        audio=AudioReference(audio_ref_id="AUD-SYN-004"),
    )
    provider = MockSTTProvider(
        fixtures={
            "AUD-SYN-004": STTResult(text="Synthetic spoken addendum.", language="en", confidence=0.8)
        }
    )
    bundle = build_evidence_bundle(envelope, stt_provider=provider)

    assert bundle.transcript is not None
    assert "Synthetic portal complaint text" in bundle.transcript.text
    assert "Synthetic spoken addendum" in bundle.transcript.text
    assert "test_category" in bundle.transcript.text


# ---------------------------------------------------------------------------
# 13. Invalid InputEnvelope
# ---------------------------------------------------------------------------
def test_invalid_envelope_type_raises_fusion_error():
    with pytest.raises(FusionError) as exc_info:
        build_evidence_bundle({"not": "an envelope"})  # type: ignore[arg-type]
    assert exc_info.value.code == "INVALID_INPUT"


def test_constructing_envelope_without_required_fields_is_rejected_upstream():
    # Layer 0's own pydantic validation must reject this before it could
    # ever reach Layer 1 — confirms Layer 1 does not need to re-implement
    # Layer 0's structural checks.
    with pytest.raises(ValidationError):
        InputEnvelope(channel=Channel.CHATBOT, modalities=[])  # missing case_id, empty modalities


# ---------------------------------------------------------------------------
# 14. Safe error handling (no raw exceptions/PII leak through FusionError)
# ---------------------------------------------------------------------------
def test_stt_failure_is_wrapped_safely_without_leaking_internals():
    class BoomProvider:
        def transcribe(self, audio):
            raise RuntimeError("raw provider secret: SUPER_SECRET_TOKEN_XYZ")

    envelope = _envelope(
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        text=None,
        audio=AudioReference(audio_ref_id="AUD-SYN-005"),
    )

    with pytest.raises(FusionError) as exc_info:
        build_evidence_bundle(envelope, stt_provider=BoomProvider())  # type: ignore[arg-type]

    assert exc_info.value.code == "STT_FAILED"
    assert "SUPER_SECRET_TOKEN_XYZ" not in exc_info.value.message
    assert "SUPER_SECRET_TOKEN_XYZ" not in str(exc_info.value.details)


def test_redaction_failure_is_wrapped_safely():
    class BoomRedactor:
        def redact(self, text, language=None):
            raise RuntimeError(f"internal failure on raw text: {text}")

    envelope = _envelope(text=TextContent(body="synthetic sensitive content"))

    with pytest.raises(FusionError) as exc_info:
        build_evidence_bundle(envelope, pii_redactor=BoomRedactor())  # type: ignore[arg-type]

    assert exc_info.value.code == "MODEL_FAILED"
    assert "synthetic sensitive content" not in exc_info.value.message
    assert "synthetic sensitive content" not in str(exc_info.value.details)


def test_evidence_bundle_never_reveals_case_id_mismatch():
    # Basic integrity check: the bundle must trace back to the originating
    # case/input, never to a different one.
    envelope = _envelope(case_id="CASE-ABCDEF01")
    bundle = build_evidence_bundle(envelope)
    assert bundle.case_id == "CASE-ABCDEF01"
    assert envelope.input_id in bundle.source_input_ids
