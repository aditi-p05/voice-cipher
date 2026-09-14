from __future__ import annotations

from backend.ingestion.input_envelope import AudioReference, InputEnvelope
from backend.models.enums import Channel, Modality
from backend.orchestration.nodes import Providers
from backend.orchestration.voice_session import (
    InMemoryVoiceSessionStore,
    end_voice_session,
    ingest_voice_chunk,
)
from services.fusion.errors import FusionError
from services.fusion.fusion_service import build_evidence_bundle
from services.fusion.stt.base import STTResult
from services.fusion.stt.mock_provider import MockSTTProvider


def chunk_envelope(call_id: str, audio_ref_id: str, sequence_number: int, case_id="CASE-VOICE-SESS"):
    return InputEnvelope(
        case_id=case_id,
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        audio=AudioReference(
            audio_ref_id=audio_ref_id,
            call_id=call_id,
            is_chunk=True,
            sequence_number=sequence_number,
        ),
    )


def _stt_fusion_fn(fixtures: dict[str, str]):
    """Build a fusion_fn that transcribes via a MockSTTProvider pre-loaded
    with synthetic fixtures, so tests can simulate 'voice with content'
    without pretending real speech was transcribed."""
    provider = MockSTTProvider(
        fixtures={
            ref_id: STTResult(text=text, language="en", confidence=0.9, segments=[])
            for ref_id, text in fixtures.items()
        }
    )
    return lambda envelope: build_evidence_bundle(envelope, stt_provider=provider)


def test_first_chunk_alone_produces_a_scored_case():
    store = InMemoryVoiceSessionStore()
    fusion_fn = _stt_fusion_fn({"AR-1": "I am scared of what might happen."})
    providers = Providers(fusion_fn=fusion_fn)

    case = ingest_voice_chunk(chunk_envelope("CALL-1", "AR-1", 0), store=store, providers=providers)

    assert case["evidence_available"] is True
    assert case["chunks_processed"] == 1
    assert case["svi_result"] is not None


def test_second_chunk_rescoring_sees_cumulative_evidence_not_just_itself():
    store = InMemoryVoiceSessionStore()
    fusion_fn = _stt_fusion_fn(
        {
            "AR-1": "Everything has been fine lately.",
            "AR-2": "He said he would hurt me if I told anyone.",
        }
    )
    providers = Providers(fusion_fn=fusion_fn)

    first = ingest_voice_chunk(chunk_envelope("CALL-2", "AR-1", 0, case_id="CASE-VOICE-SESS2"), store=store, providers=providers)
    second = ingest_voice_chunk(chunk_envelope("CALL-2", "AR-2", 1, case_id="CASE-VOICE-SESS2"), store=store, providers=providers)

    assert first["chunks_processed"] == 1
    assert second["chunks_processed"] == 2
    # The threat marker from chunk 2 must show up in the merged evidence,
    # not be scored in isolation from chunk 1's (calmer) content.
    marker_types = {m["type"] for m in second["evidence_bundle"]["markers"]}
    assert "threat" in marker_types
    assert "Everything has been fine" in second["evidence_bundle"]["transcript"]["text"]
    assert "he would hurt me" in second["evidence_bundle"]["transcript"]["text"]


def test_a_critical_marker_in_an_early_chunk_is_not_diluted_by_later_calm_chunks():
    store = InMemoryVoiceSessionStore()
    fusion_fn = _stt_fusion_fn(
        {
            "AR-1": "I want to end my life.",
            "AR-2": "Sorry, ignore that, everything is actually fine now.",
        }
    )
    providers = Providers(fusion_fn=fusion_fn)

    first = ingest_voice_chunk(chunk_envelope("CALL-3", "AR-1", 0, case_id="CASE-VOICE-SESS3"), store=store, providers=providers)
    second = ingest_voice_chunk(chunk_envelope("CALL-3", "AR-2", 1, case_id="CASE-VOICE-SESS3"), store=store, providers=providers)

    assert first["risk_tier"] == "CRITICAL"
    # The self_harm marker from chunk 1 must still be present after chunk 2 --
    # the merge unions markers, it does not let a calmer later chunk erase them.
    assert second["risk_tier"] == "CRITICAL"
    assert second["support_decision"]["support_enabled"] is False


def test_fusion_failure_on_one_chunk_does_not_break_the_session():
    def flaky_fusion(envelope):
        if envelope.audio.audio_ref_id == "AR-BAD":
            raise FusionError("MODEL_FAILED", "Evidence fusion failed unexpectedly.")
        return build_evidence_bundle(envelope, stt_provider=MockSTTProvider())

    store = InMemoryVoiceSessionStore()
    providers = Providers(fusion_fn=flaky_fusion)

    bad = ingest_voice_chunk(chunk_envelope("CALL-4", "AR-BAD", 0, case_id="CASE-VOICE-SESS4"), store=store, providers=providers)
    assert bad["evidence_available"] is False
    assert any(e["code"] == "FUSION_FAILED" for e in bad["errors"])

    good = ingest_voice_chunk(chunk_envelope("CALL-4", "AR-GOOD", 1, case_id="CASE-VOICE-SESS4"), store=store, providers=providers)
    assert good["evidence_available"] is True
    assert good["chunks_processed"] == 1  # only the successful chunk counted
    # The earlier chunk's failure is still visible in this case's error history.
    assert any(e["code"] == "FUSION_FAILED" for e in good["errors"])


def test_end_voice_session_clears_state_for_that_call_only():
    store = InMemoryVoiceSessionStore()
    fusion_fn = _stt_fusion_fn({"AR-1": "hello"})
    providers = Providers(fusion_fn=fusion_fn)

    ingest_voice_chunk(chunk_envelope("CALL-5", "AR-1", 0, case_id="CASE-VOICE-SESS5"), store=store, providers=providers)
    assert store.get("CALL-5") is not None

    end_voice_session("CALL-5", store=store)
    assert store.get("CALL-5") is None


def test_missing_call_id_is_rejected():
    import pytest

    envelope = InputEnvelope(
        case_id="CASE-X",
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO],
        audio=AudioReference(audio_ref_id="AR-1"),  # no call_id
    )
    with pytest.raises(ValueError):
        ingest_voice_chunk(envelope, store=InMemoryVoiceSessionStore())
