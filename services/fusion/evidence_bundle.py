"""
Layer 1 (Fusion / Privacy) — EvidenceBundle schema.

This is the shared object defined in CONTRACTS.md section 6.2. It is the
ONLY thing Members 3 and 4 depend on; they must never import raw source
text, raw audio, or Layer 0 internals to get "real" evidence.

Hard privacy rule (CONTRACTS.md section 6.2 + this module's docstring):
`transcript.text`, when present, MUST already be redacted before an
EvidenceBundle is constructed. This module does not redact; it only
defines the shape. Redaction happens in redaction/ and is enforced by
fusion_service.build_evidence_bundle before this model is ever built.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_BUNDLE_SCHEMA_VERSION = "1.0.0"


class TranscriptSegment(BaseModel):
    """A single time-aligned (or sequence-aligned) piece of a transcript."""

    text: str
    start_seconds: Optional[float] = Field(default=None, ge=0)
    end_seconds: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class Transcript(BaseModel):
    """
    Redacted transcript produced from text and/or STT output.

    `text` must already be PII-redacted by the time it reaches this model.
    `confidence` is an overall 0-1 confidence for the transcript as a whole
    (STT confidence for audio-derived transcripts; 1.0 for direct text).
    """

    text: str
    language: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    segments: list[TranscriptSegment] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class Marker(BaseModel):
    """
    An explainable, non-diagnostic vulnerability/safety marker.

    `type` is the marker category (e.g. "threat", "fear", "self_harm").
    `value` is a short explainable label/snippet for *why* it fired
    (e.g. "retaliation"), never a diagnosis and never raw PII.
    `confidence` is 0-1.
    """

    type: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class Sentiment(BaseModel):
    """Dimensional sentiment/arousal representation, not a diagnosis."""

    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(extra="forbid")


class VoiceFeatures(BaseModel):
    """
    Optional acoustic features. The system MUST work with
    `voice_features = None`; nothing downstream may require this.
    """

    pitch_hz_mean: Optional[float] = Field(default=None, ge=0)
    pitch_hz_stdev: Optional[float] = Field(default=None, ge=0)
    jitter: Optional[float] = Field(default=None, ge=0)
    shimmer: Optional[float] = Field(default=None, ge=0)
    pause_frequency: Optional[float] = Field(default=None, ge=0)
    speech_rate_wpm: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(extra="forbid")


class EvidenceBundle(BaseModel):
    """
    Layer 1's redacted, multimodal evidence output (CONTRACTS.md 6.2).

    Producer: Member 2 (Fusion / Privacy).
    Consumers: Member 3 (SVI/Risk), Member 4 (Service/RAG).

    Downstream members must depend on this shape only — never on raw
    complaint text, raw audio, or services/fusion internals.
    """

    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION
    case_id: str
    source_input_ids: list[str] = Field(default_factory=list)

    transcript: Optional[Transcript] = None
    markers: list[Marker] = Field(default_factory=list)
    sentiment: Optional[Sentiment] = None
    voice_features: Optional[VoiceFeatures] = None

    pii_redacted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="forbid")
