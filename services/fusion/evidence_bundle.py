"""
services/fusion/evidence_bundle.py

*** PLACEHOLDER CONTRACT — NOT the real Member-2-owned module. ***

This file did NOT ship in the zip you gave me (voice-cipher-gap1...).
Every other module (evidence_merge.py, run.py, state.py, nodes.py via
fusion_adapter) imports `from services.fusion.evidence_bundle import
EvidenceBundle, Marker, Sentiment, Transcript` and none of them work
without it, so I wrote a minimal version that matches exactly what the
rest of the codebase already assumes about its shape:

  - EvidenceBundle is a pydantic model (run.py calls
    `evidence_bundle.model_dump(mode="json")`)
  - EvidenceBundle(case_id, source_input_ids, transcript, markers,
    sentiment, voice_features, pii_redacted) — these exact kwargs are
    used in evidence_merge.py's `merge_evidence_bundles`
  - Transcript(text, language, confidence, segments) — same source
  - Sentiment(valence, arousal) — same source
  - Marker is only ever appended/unioned, never field-accessed
    elsewhere in the code you shared, so its fields below are my
    best guess at what Layer 2 (SVI) will want to read.

ACTION ITEM: the moment your Member 2 (fusion service owner) shares
their real services/fusion/evidence_bundle.py, replace this file with
theirs and re-check backend/orchestration/adapters/fusion_adapter.py
still builds a matching EvidenceBundle. Field names were chosen to
already match usage elsewhere, so the swap should mostly be additive.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TranscriptSegment(BaseModel):
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    speaker: Optional[str] = None
    text: str = ""
    language: Optional[str] = None


class Transcript(BaseModel):
    text: str = ""
    language: Optional[str] = None
    confidence: float = 1.0
    segments: list[TranscriptSegment] = Field(default_factory=list)


class Sentiment(BaseModel):
    # valence: -1 (very negative) .. +1 (very positive)
    # arousal:  0 (calm)          ..  1 (highly agitated)
    valence: float = 0.0
    arousal: float = 0.0


class Marker(BaseModel):
    """One explicit, rule-based piece of evidence found in a transcript
    (NOT a tone/emotion inference — those live in `Sentiment` instead).
    Kept deliberately explainable: an operator or SVI should be able to
    see exactly which phrase tripped this marker and why."""

    # `type`/`value`/`confidence` is the contract consumed by the SVI rule
    # and feature layers.  The legacy names below remain available because
    # the earlier orchestration fusion adapter still produces them.
    type: str
    value: str = ""
    confidence: float = 1.0
    marker_type: str | None = None
    severity: str | None = None
    matched_text: str | None = None
    source: str = "transcript_rule_based"  # vs. e.g. "gemini_tone" if ever added
    segment_start_seconds: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_shape(cls, data):
        """Accept the old adapter shape while exposing the shared SVI shape."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        legacy_type = values.get("marker_type")
        if not values.get("type") and legacy_type:
            values["type"] = {
                "weapon_mention": "weapon",
                "abuse_disclosure": "violence",
                "explicit_threat": "threat",
            }.get(legacy_type, legacy_type)
        if "value" not in values and values.get("matched_text") is not None:
            values["value"] = values["matched_text"]
        if "confidence" not in values:
            values["confidence"] = {
                "critical": 0.95,
                "high": 0.85,
                "medium": 0.65,
            }.get(values.get("severity"), 1.0)
        if not values.get("marker_type") and values.get("type"):
            values["marker_type"] = values["type"]
        if values.get("matched_text") is None and "value" in values:
            values["matched_text"] = values["value"]
        return values


class VoiceFeatures(BaseModel):
    """Prosody/voice-stress signal. Populated by the separate
    voice-stress-analysis engine (retained from Detox.ai — /analyze-voice),
    not by this Gemini transcription adapter. Left optional/empty here."""

    pitch_hz_mean: Optional[float] = None
    pitch_hz_stdev: Optional[float] = None
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    pause_frequency: Optional[float] = None
    # Kept for compatibility with earlier feature-extractor fixtures.
    pitch_mean_hz: Optional[float] = None
    speaking_rate_wpm: Optional[float] = None


class EvidenceBundle(BaseModel):
    schema_version: str = "1.0.0"
    case_id: str
    source_input_ids: list[str] = Field(default_factory=list)
    transcript: Optional[Transcript] = None
    markers: list[Marker] = Field(default_factory=list)
    sentiment: Optional[Sentiment] = None
    voice_features: Optional[VoiceFeatures] = None
    pii_redacted: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
