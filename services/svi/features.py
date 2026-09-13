"""
Layer 2 (SVI / Risk) — feature extraction for the ML scorer.

Turns an EvidenceBundle into the flat, numeric feature dict
services/svi/ml/*.MLScorer implementations expect (see
services/svi/ml/heuristic_scorer.py's weight keys). Pure and defensive:
never reads or returns raw transcript text, raw audio, or any other raw
PII -- only counts, confidences, and continuous acoustic/sentiment
signals already present on the (already-redacted) EvidenceBundle.

Missing optional evidence (no sentiment, no voice_features, no
transcript, no markers) never raises here -- it simply yields the
corresponding features as 0.0/absent, exactly like
services/svi/rules.py's handling of missing markers.
"""

from __future__ import annotations

from services.fusion import EvidenceBundle


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_features(evidence_bundle: EvidenceBundle) -> dict[str, float]:
    markers = list(getattr(evidence_bundle, "markers", None) or [])
    sentiment = getattr(evidence_bundle, "sentiment", None)
    voice_features = getattr(evidence_bundle, "voice_features", None)
    transcript = getattr(evidence_bundle, "transcript", None)

    confidences = [float(m.confidence) for m in markers]
    marker_types = {m.type for m in markers}

    valence = _clamp(float(sentiment.valence), -1.0, 1.0) if sentiment is not None else 0.0
    arousal = _clamp(float(sentiment.arousal), 0.0, 1.0) if sentiment is not None else 0.0

    return {
        "marker_count": float(len(markers)),
        "max_marker_confidence": max(confidences) if confidences else 0.0,
        "distinct_marker_types": float(len(marker_types)),
        "sentiment_valence": valence,
        "sentiment_arousal": arousal,
        "voice_pitch_stdev": float(voice_features.pitch_hz_stdev or 0.0) if voice_features else 0.0,
        "voice_jitter": float(voice_features.jitter or 0.0) if voice_features else 0.0,
        "voice_shimmer": float(voice_features.shimmer or 0.0) if voice_features else 0.0,
        "voice_pause_frequency": float(voice_features.pause_frequency or 0.0) if voice_features else 0.0,
        "transcript_confidence": float(transcript.confidence) if transcript is not None else 0.0,
        "has_transcript": 1.0 if transcript is not None else 0.0,
    }
