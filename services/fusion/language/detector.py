"""
Layer 1 (Fusion / Privacy) — language detection.

`language_hint` on the InputEnvelope (Layer 0) is only ever a hint — it is
declared by the client/channel and never verified (CONTRACTS.md 6.1, 10).
Whenever actual text is available, this module performs real detection
and that result — not the hint — is what gets preserved on the
EvidenceBundle's transcript. The hint is used only as a fallback when
detection is impossible (e.g. no text, or text too short/ambiguous).

Uses `langdetect`, a small pure-Python library with no model downloads,
so tests and demos never need network access or heavy assets.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0  # deterministic results across runs
    _LANGDETECT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if dependency missing
    _LANGDETECT_AVAILABLE = False

MIN_TEXT_LENGTH_FOR_DETECTION = 3


@dataclass(frozen=True)
class LanguageDetectionResult:
    language: str | None
    confidence: float
    source: str  # "detected" | "hint" | "unknown"


def detect_language(
    text: str | None, language_hint: str | None = None
) -> LanguageDetectionResult:
    """
    Detect language from `text` when possible; otherwise gracefully fall
    back to `language_hint`; otherwise return an explicit "unknown" result
    rather than guessing.
    """
    cleaned = (text or "").strip()

    if cleaned and len(cleaned) >= MIN_TEXT_LENGTH_FOR_DETECTION and _LANGDETECT_AVAILABLE:
        try:
            candidates = detect_langs(cleaned)
            if candidates:
                top = candidates[0]
                return LanguageDetectionResult(
                    language=top.lang, confidence=float(top.prob), source="detected"
                )
        except LangDetectException:
            pass  # fall through to hint/unknown below

    if language_hint:
        return LanguageDetectionResult(language=language_hint, confidence=0.0, source="hint")

    return LanguageDetectionResult(language=None, confidence=0.0, source="unknown")
