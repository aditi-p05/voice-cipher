"""
Layer 1 (Fusion / Privacy) — explainable vulnerability/safety markers.

Deliberately a small, deterministic, keyword/phrase-based extractor:
- Explainable: every marker's `value` is the literal cue phrase that
  fired, so Member 3/6 (and an auditor) can see exactly why it appeared.
- Not a diagnosis: these are lexical safety/vulnerability signals only.
  Nothing here claims or implies a clinical or psychiatric assessment.
- No model download required for the demo; a learned classifier can
  later populate the same `Marker` shape (services/fusion/evidence_bundle.py)
  without changing what Member 3 consumes.

Confidence is deliberately coarse (per-keyword weight, capped at 0.95) —
it signals relative lexical strength, not a calibrated probability.
"""

from __future__ import annotations

import re

MarkerType = str

# type -> list of (cue phrase, confidence). Phrases are matched as
# case-insensitive whole-word/phrase substrings against redacted text.
# Extend deliberately; keep cues in plain, explainable language.
_MARKER_LEXICON: dict[MarkerType, list[tuple[str, float]]] = {
    "threat": [
        ("i will kill", 0.9),
        ("i'll kill", 0.9),
        ("threatened to", 0.75),
        ("said he would hurt", 0.8),
        ("said she would hurt", 0.8),
        ("if you tell anyone", 0.7),
    ],
    "fear": [
        ("i am scared", 0.7),
        ("i'm scared", 0.7),
        ("i am afraid", 0.7),
        ("i'm afraid", 0.7),
        ("i am terrified", 0.8),
        ("living in fear", 0.75),
    ],
    "retaliation": [
        ("if i report", 0.7),
        ("if i complain", 0.7),
        ("he said he would come back", 0.7),
        ("she said she would come back", 0.7),
        ("get back at me", 0.65),
    ],
    "unsafe": [
        ("not safe", 0.6),
        ("no longer safe", 0.7),
        ("unsafe at home", 0.75),
    ],
    "coercion": [
        ("forced me", 0.75),
        ("made me do", 0.6),
        ("threatened to expose", 0.75),
        ("blackmail", 0.8),
    ],
    "violence": [
        ("hit me", 0.85),
        ("beat me", 0.85),
        ("assaulted", 0.85),
        ("attacked me", 0.8),
        ("choked me", 0.9),
    ],
    "self_harm": [
        ("want to end my life", 0.95),
        ("want to die", 0.9),
        ("hurt myself", 0.85),
        ("kill myself", 0.95),
        ("suicide", 0.9),
    ],
    "weapon": [
        ("with a knife", 0.85),
        ("with a gun", 0.9),
        ("had a weapon", 0.8),
        ("pointed a gun", 0.9),
    ],
    "immediate_danger": [
        ("he is outside right now", 0.9),
        ("she is outside right now", 0.9),
        ("right now", 0.4),
        ("happening right now", 0.7),
        ("he is here now", 0.9),
    ],
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_markers(text: str | None) -> list[dict]:
    """
    Return a list of `{"type", "value", "confidence"}` dicts (matching
    services.fusion.evidence_bundle.Marker) found in `text`.

    `text` is expected to already be redacted; this function only reads
    the shape of the language, not any PII within it. Returns an empty
    list for empty/None input rather than raising.
    """
    if not text:
        return []

    normalized = _normalize(text)
    markers: list[dict] = []

    for marker_type, cues in _MARKER_LEXICON.items():
        for phrase, confidence in cues:
            if phrase in normalized:
                markers.append(
                    {"type": marker_type, "value": phrase, "confidence": min(confidence, 0.95)}
                )

    return markers
