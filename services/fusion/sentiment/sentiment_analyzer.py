"""
Layer 1 (Fusion / Privacy) — sentiment/arousal representation.

Deterministic lexicon-based scorer, matching the "if a sophisticated
model is unavailable, use a deterministic/mock implementation for
development rather than breaking the pipeline" instruction. Produces a
`Sentiment(valence, arousal)` pair (services.fusion.evidence_bundle.Sentiment)
rather than a single polarity score, since downstream risk logic needs
both dimensions (CONTRACTS.md 6.2 example).

Not a clinical mood/affect assessment — a coarse dimensional signal only.
"""

from __future__ import annotations

import re

_NEGATIVE_WORDS = {
    "scared", "afraid", "terrified", "threatened", "hurt", "unsafe", "fear",
    "crying", "helpless", "trapped", "abused", "assaulted", "violent",
    "hopeless", "worthless", "alone", "panic",
}
_POSITIVE_WORDS = {
    "safe", "calm", "relieved", "grateful", "supported", "hopeful", "better",
    "reassured", "protected",
}
_HIGH_AROUSAL_WORDS = {
    "screaming", "panic", "shaking", "terrified", "urgent", "immediately",
    "right now", "help me", "can't breathe", "shouting",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def analyze_sentiment(text: str | None) -> dict:
    """
    Return `{"valence": float in [-1, 1], "arousal": float in [0, 1]}`
    matching services.fusion.evidence_bundle.Sentiment.

    Empty/None text returns a neutral-low reading rather than raising,
    so the fusion pipeline never breaks on missing content.
    """
    if not text:
        return {"valence": 0.0, "arousal": 0.0}

    normalized = _normalize(text)
    words = re.findall(r"[a-z']+", normalized)

    neg_hits = sum(1 for w in words if w in _NEGATIVE_WORDS)
    pos_hits = sum(1 for w in words if w in _POSITIVE_WORDS)
    arousal_hits = sum(1 for phrase in _HIGH_AROUSAL_WORDS if phrase in normalized)

    total_sentiment_hits = neg_hits + pos_hits
    if total_sentiment_hits == 0:
        valence = 0.0
    else:
        valence = (pos_hits - neg_hits) / total_sentiment_hits

    # Exclamation marks and repeated punctuation are a light additional
    # arousal cue on top of lexical hits.
    punctuation_intensity = min(normalized.count("!"), 3) * 0.1
    arousal = _clamp(arousal_hits * 0.25 + punctuation_intensity, 0.0, 1.0)

    return {"valence": round(_clamp(valence, -1.0, 1.0), 2), "arousal": round(arousal, 2)}
