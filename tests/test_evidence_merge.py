from __future__ import annotations

import pytest

from backend.orchestration.evidence_merge import merge_evidence_bundles
from services.fusion.evidence_bundle import (
    EvidenceBundle,
    Marker,
    Sentiment,
    Transcript,
    TranscriptSegment,
    VoiceFeatures,
)


def bundle(case_id="CASE-V1", text=None, sentiment=None, markers=None, redacted=True, voice_features=None):
    transcript = Transcript(text=text, segments=[TranscriptSegment(text=text)]) if text else None
    return EvidenceBundle(
        case_id=case_id,
        source_input_ids=[f"IN-{id(object())}"],
        transcript=transcript,
        markers=markers or [],
        sentiment=Sentiment(**sentiment) if sentiment else None,
        voice_features=VoiceFeatures(**voice_features) if voice_features else None,
        pii_redacted=redacted if transcript else False,
    )


def test_merge_requires_at_least_one_bundle():
    with pytest.raises(ValueError):
        merge_evidence_bundles([])


def test_merge_rejects_mismatched_case_ids():
    with pytest.raises(ValueError):
        merge_evidence_bundles([bundle(case_id="A"), bundle(case_id="B")])


def test_transcript_concatenated_in_order():
    merged = merge_evidence_bundles([bundle(text="first chunk."), bundle(text="second chunk.")])
    assert merged.transcript.text == "first chunk. second chunk."
    assert len(merged.transcript.segments) == 2


def test_markers_are_unioned_not_overwritten():
    m1 = Marker(type="fear", value="i am scared", confidence=0.7)
    m2 = Marker(type="threat", value="threatened to", confidence=0.75)
    merged = merge_evidence_bundles([bundle(markers=[m1]), bundle(markers=[m2])])
    types = {m.type for m in merged.markers}
    assert types == {"fear", "threat"}


def test_sentiment_takes_most_concerning_reading_not_average():
    # One calm-ish chunk, one severely negative/activated chunk. A naive
    # average would water this down; the merge must not do that.
    calm = bundle(sentiment={"valence": 0.1, "arousal": 0.1})
    severe = bundle(sentiment={"valence": -0.9, "arousal": 0.95})
    merged = merge_evidence_bundles([calm, severe])
    assert merged.sentiment.valence == -0.9  # most negative wins
    assert merged.sentiment.arousal == 0.95  # highest arousal wins


def test_voice_features_takes_latest_non_null():
    early = bundle(voice_features=None)
    later = bundle(voice_features={"pitch_hz_mean": 210.0})
    merged = merge_evidence_bundles([early, later])
    assert merged.voice_features.pitch_hz_mean == 210.0


def test_pii_redacted_true_when_all_text_chunks_were_redacted():
    merged = merge_evidence_bundles([bundle(text="a", redacted=True), bundle(text="b", redacted=True)])
    assert merged.pii_redacted is True


def test_pii_redacted_false_if_any_text_chunk_was_not_redacted():
    merged = merge_evidence_bundles([bundle(text="a", redacted=True), bundle(text="b", redacted=False)])
    assert merged.pii_redacted is False


def test_pii_redacted_true_when_no_chunk_has_any_text_at_all():
    merged = merge_evidence_bundles([bundle(text=None), bundle(text=None)])
    assert merged.pii_redacted is True


def test_source_input_ids_accumulate_across_all_chunks():
    merged = merge_evidence_bundles([bundle(), bundle(), bundle()])
    assert len(merged.source_input_ids) == 3
