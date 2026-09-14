"""
Layer 4B (Orchestration) — merge multiple EvidenceBundles.

Used when one case accumulates evidence across several separate Layer 1
runs -- today, specifically repeated /voice/chunk interactions for one
call (see voice_session.py) -- into one cumulative bundle for SVI /
Service / Support.

Design decision -- safety-conservative merge, not an average:
Sentiment is merged toward whichever individual chunk's reading was MOST
concerning (most negative valence, highest arousal), never averaged.
Averaging would let a brief but severe spike (e.g. one chunk showing
acute fear) get diluted by several calmer chunks around it -- exactly
the kind of signal a safety-critical triage system must not miss. The
same reasoning applies to markers: a marker that fired in any one chunk
is kept, not weighted down by chunks where it didn't fire.
"""

from __future__ import annotations

from typing import Sequence

from services.fusion.evidence_bundle import (
    EvidenceBundle,
    Marker,
    Sentiment,
    Transcript,
)


def merge_evidence_bundles(bundles: Sequence[EvidenceBundle]) -> EvidenceBundle:
    if not bundles:
        raise ValueError("merge_evidence_bundles requires at least one bundle")

    case_ids = {b.case_id for b in bundles}
    if len(case_ids) > 1:
        raise ValueError(f"All bundles being merged must share one case_id, got: {case_ids}")
    case_id = bundles[0].case_id

    source_input_ids = [input_id for b in bundles for input_id in b.source_input_ids]

    # Transcript: concatenate in arrival order; keep every chunk's segments.
    text_parts: list[str] = []
    segments = []
    languages: list[str] = []
    min_confidence = 1.0
    for b in bundles:
        if b.transcript is not None:
            if b.transcript.text:
                text_parts.append(b.transcript.text)
            segments.extend(b.transcript.segments)
            if b.transcript.language:
                languages.append(b.transcript.language)
            min_confidence = min(min_confidence, b.transcript.confidence)

    transcript = None
    if text_parts:
        transcript = Transcript(
            text=" ".join(text_parts),
            language=languages[-1] if languages else None,
            confidence=min_confidence,
            segments=segments,
        )

    # Markers: union across all chunks -- a marker that fired once still matters.
    markers: list[Marker] = [m for b in bundles for m in b.markers]

    # Sentiment: most-concerning individual reading wins; never averaged away.
    sentiments = [b.sentiment for b in bundles if b.sentiment is not None]
    sentiment = None
    if sentiments:
        sentiment = Sentiment(
            valence=min(s.valence for s in sentiments),
            arousal=max(s.arousal for s in sentiments),
        )

    # Voice features: most recent non-null reading (later chunks reflect
    # more speech to measure prosody from than earlier, shorter ones).
    voice_features = None
    for b in bundles:
        if b.voice_features is not None:
            voice_features = b.voice_features

    # pii_redacted: true only if every chunk that actually had text content
    # was itself redacted. A chunk with no text at all doesn't count against
    # this (nothing existed to redact), matching Layer 1's own semantics.
    chunks_with_text = [b for b in bundles if b.transcript is not None]
    pii_redacted = all(b.pii_redacted for b in chunks_with_text) if chunks_with_text else True

    return EvidenceBundle(
        case_id=case_id,
        source_input_ids=source_input_ids,
        transcript=transcript,
        markers=markers,
        sentiment=sentiment,
        voice_features=voice_features,
        pii_redacted=pii_redacted,
    )
