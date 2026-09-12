"""
Layer 1 (Fusion / Privacy) — public adapter.

    InputEnvelope -> build_evidence_bundle(envelope) -> EvidenceBundle

This is the ONLY function Members 3 and 4 (and any orchestration glue)
should call. Everything else in services/fusion is an implementation
detail reachable through this adapter's dependency-injected providers,
not through direct imports of fusion internals.

Design summary (see CONTRACTS.md 6.2 and the Layer 1 task brief):
- Handles text-only, audio-only, audio+text, text+structured_data, and
  portal-with-audio inputs (multimodal fusion, responsibility 8).
- Produces REDACTED text only; nothing here ever puts raw complaint
  text/audio into the returned EvidenceBundle (privacy rule).
- Degrades gracefully: missing voice features, missing transcript, or an
  unknown language never raise — they simply produce `None`/neutral
  fields. Only structurally invalid input or a hard provider failure
  raises FusionError.
- Never logs raw text/PII; only safe, non-content fields are ever logged.
"""

from __future__ import annotations

from backend.core.logging import log_event
from backend.ingestion.input_envelope import InputEnvelope
from backend.models.enums import Modality
from services.fusion.errors import INVALID_INPUT, MODEL_FAILED, STT_FAILED, FusionError
from services.fusion.evidence_bundle import (
    EvidenceBundle,
    Marker,
    Sentiment,
    Transcript,
    TranscriptSegment,
    VoiceFeatures,
)
from services.fusion.language.detector import detect_language
from services.fusion.markers.marker_extractor import extract_markers
from services.fusion.redaction.base import PiiRedactor
from services.fusion.redaction.regex_redactor import RegexPiiRedactor
from services.fusion.sentiment.sentiment_analyzer import analyze_sentiment
from services.fusion.stt.base import SpeechToTextProvider, STTResult
from services.fusion.stt.mock_provider import MockSTTProvider
from services.fusion.text_processing import extract_source_text
from services.fusion.voice_features.extractor import (
    NullVoiceFeatureExtractor,
    VoiceFeatureExtractor,
)


def _run_stt(envelope: InputEnvelope, provider: SpeechToTextProvider) -> STTResult | None:
    if Modality.AUDIO not in envelope.modalities or envelope.audio is None:
        return None
    try:
        return provider.transcribe(envelope.audio)
    except FusionError:
        raise
    except Exception as exc:  # never leak a raw provider exception upward
        raise FusionError(
            STT_FAILED,
            "Speech-to-text processing failed.",
            details={"input_id": envelope.input_id},
        ) from exc


def _redact(redactor: PiiRedactor, text: str, language: str | None) -> tuple[str, bool]:
    if not text:
        return "", False
    try:
        result = redactor.redact(text, language=language)
    except FusionError:
        raise
    except Exception as exc:
        raise FusionError(
            MODEL_FAILED, "PII redaction failed.", details={"stage": "redaction"}
        ) from exc
    return result.redacted_text, True


def build_evidence_bundle(
    envelope: InputEnvelope,
    *,
    stt_provider: SpeechToTextProvider | None = None,
    pii_redactor: PiiRedactor | None = None,
    voice_feature_extractor: VoiceFeatureExtractor | None = None,
) -> EvidenceBundle:
    """
    Build a privacy-safe, redacted EvidenceBundle from a Layer 0
    InputEnvelope.

    Providers default to lightweight, no-download implementations so this
    function works out of the box in tests and demos:
      - stt_provider: MockSTTProvider() — see stt/mock_provider.py for
        why it never fabricates a transcript for audio without a
        registered synthetic fixture.
      - pii_redactor: RegexPiiRedactor() — deterministic, stdlib-only.
      - voice_feature_extractor: NullVoiceFeatureExtractor() — always
        returns None, matching "voice_features may be null".

    Raises FusionError (INVALID_INPUT, STT_FAILED, MODEL_FAILED) on
    genuine failures. Never raises for merely-absent optional evidence
    (no transcript, no voice features, unknown language).
    """
    if not isinstance(envelope, InputEnvelope):
        raise FusionError(INVALID_INPUT, "build_evidence_bundle requires a valid InputEnvelope.")

    stt_provider = stt_provider or MockSTTProvider()
    pii_redactor = pii_redactor or RegexPiiRedactor()
    voice_feature_extractor = voice_feature_extractor or NullVoiceFeatureExtractor()

    try:
        # --- 1. Gather raw (in-memory only, never logged) text sources ---
        text_source = extract_source_text(envelope)  # chatbot/portal text + structured fields
        stt_result = _run_stt(envelope, stt_provider)

        stt_text = stt_result.text if stt_result and stt_result.text else ""
        combined_raw_text = " ".join(p for p in (text_source, stt_text) if p).strip()

        # --- 2. Language: real text wins over the unverified hint ---
        lang_result = detect_language(combined_raw_text or None, envelope.language_hint)
        language = lang_result.language  # gracefully None if truly undetected

        # --- 3. Redact before anything downstream can see raw text ---
        redacted_text, was_redacted = _redact(pii_redactor, combined_raw_text, language)

        for segment in stt_result.segments if stt_result else []:
            _redact(pii_redactor, segment.text, language)  # validate; per-segment text below

        # --- 4. Transcript (None when there is genuinely no text evidence) ---
        transcript = None
        if redacted_text:
            confidence = stt_result.confidence if (stt_result and stt_text) else 1.0
            segments = []
            if stt_result and stt_result.segments:
                for seg in stt_result.segments:
                    seg_redacted, _ = _redact(pii_redactor, seg.text, language)
                    segments.append(
                        TranscriptSegment(
                            text=seg_redacted,
                            start_seconds=seg.start_seconds,
                            end_seconds=seg.end_seconds,
                        )
                    )
            transcript = Transcript(
                text=redacted_text,
                language=language,
                confidence=confidence,
                segments=segments,
            )

        # --- 5. Explainable markers + sentiment, over redacted text only ---
        markers = [Marker(**m) for m in extract_markers(redacted_text)] if redacted_text else []
        sentiment = Sentiment(**analyze_sentiment(redacted_text)) if redacted_text else None

        # --- 6. Optional voice features; system works fine with None ---
        voice_features = None
        if Modality.AUDIO in envelope.modalities and envelope.audio is not None:
            features_dict = voice_feature_extractor.extract(envelope.audio)
            if features_dict:
                voice_features = VoiceFeatures(**features_dict)

        bundle = EvidenceBundle(
            case_id=envelope.case_id,
            source_input_ids=[envelope.input_id],
            transcript=transcript,
            markers=markers,
            sentiment=sentiment,
            voice_features=voice_features,
            pii_redacted=was_redacted,
        )

        log_event(
            "evidence_bundle_built",
            case_id=envelope.case_id,
            input_id=envelope.input_id,
            has_transcript=transcript is not None,
            marker_count=len(markers),
            has_voice_features=voice_features is not None,
            pii_redacted=was_redacted,
        )
        return bundle

    except FusionError:
        raise
    except Exception as exc:  # final safety net: never leak raw internals
        raise FusionError(
            MODEL_FAILED,
            "Evidence bundle construction failed.",
            details={"input_id": getattr(envelope, "input_id", None)},
        ) from exc
