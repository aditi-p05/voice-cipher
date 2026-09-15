"""
backend/orchestration/adapters/fusion_adapter.py

Layer 1 — Fusion adapter (Gap 1 — Gemini Audio Understanding).

Pipeline this module implements for the voice channel, per the design
agreed on before writing this:

    Caller audio (AudioReference.storage_uri)
    -> Gemini Audio Understanding   (verbatim transcript, language,
       per-segment timestamps, coarse per-segment + overall emotion cues)
    -> PII redaction                (regex fallback here -- swap in the
       real Presidio/spaCy pipeline retained from Detox.ai when ready)
    -> explicit danger/threat marker extraction over the REDACTED
       transcript text (rule-based, independent of Gemini's tone read)
    -> EvidenceBundle

Design rule: Gemini's tone/emotion read is ONE auxiliary signal (feeds
`Sentiment`) and is never by itself treated as a safety marker. Only
explicit language in the transcript (e.g. "he said he'll kill me
tonight") produces a `Marker`. SVI (Layer 2) is the only place these two
signals get combined into a risk tier -- this module only produces
evidence, it never scores risk.

For chat/portal channels (text already present, nothing to send to
Gemini) the same redaction + marker steps run directly on the given
text; `sentiment` is left null since there's no audio tone to read.

Public entrypoint (imported by nodes.py and orchestration/voice_session.py):

    run_fusion(envelope, *, fusion_fn=None, timeout_seconds=10.0)
        -> tuple[EvidenceBundle | None, dict | None]

*** Dependency note: requires the `google-genai` package
    (`pip install google-genai`) and GEMINI_API_KEY set (already wired
    in backend/orchestration/config.py). Import is done lazily inside
    _call_gemini_audio so a missing/not-yet-installed SDK doesn't break
    every other caller of this module (e.g. text-only chat/portal
    fusion, which never touches Gemini at all). ***
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from backend.ingestion.input_envelope import InputEnvelope
from backend.orchestration import config, errors
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.evidence_bundle import (
    EvidenceBundle,
    Marker,
    Sentiment,
    Transcript,
    TranscriptSegment,
)
from services.fusion.fusion_service import build_evidence_bundle

FusionFn = Callable[[InputEnvelope], EvidenceBundle]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_fusion(
    envelope: InputEnvelope,
    *,
    fusion_fn: Optional[FusionFn] = None,
    timeout_seconds: float = 10.0,
) -> tuple[Optional[EvidenceBundle], Optional[dict]]:
    """
    Fuse one InputEnvelope into an EvidenceBundle.

    Never raises: on any failure (Gemini call errors, timeout, bad
    audio, etc.) this returns (None, structured_error_dict), matching
    the "fusion fails -> no evidence, do not fabricate" philosophy
    documented in nodes.py.
    """
    fn = fusion_fn or default_fusion_fn
    try:
        bundle = run_with_timeout(lambda: fn(envelope), seconds=timeout_seconds)
        return bundle, None
    except NodeTimeoutError as exc:
        return None, errors.stage_error("fusion", str(exc), code=errors.TIMEOUT)
    except Exception as exc:  # noqa: BLE001 -- must never bubble a raw exception
        return None, errors.stage_error("fusion", f"Fusion failed: {exc}")


# ---------------------------------------------------------------------------
# Default provider
# ---------------------------------------------------------------------------


def default_fusion_fn(envelope: InputEnvelope) -> EvidenceBundle:
    """Use the Layer 1 public adapter for every modality.

    It handles text, audio-only, and multimodal envelopes without requiring
    a Gemini key for an audio reference that has no registered transcript.
    """
    return build_evidence_bundle(envelope)


def _fuse_audio(envelope: InputEnvelope) -> EvidenceBundle:
    audio = envelope.audio
    if not audio.storage_uri:
        raise ValueError("AudioReference.storage_uri is required to fetch audio bytes")

    gemini_result = _call_gemini_audio(audio.storage_uri, audio.mime_type)

    raw_text = gemini_result.get("full_text", "") or ""
    redacted_text, _ = redact_pii(raw_text)

    segments = [
        TranscriptSegment(
            start_seconds=seg.get("start_seconds"),
            end_seconds=seg.get("end_seconds"),
            speaker=seg.get("speaker"),
            text=redact_pii(seg.get("content", "") or "")[0],
            language=seg.get("language"),
        )
        for seg in gemini_result.get("segments", [])
    ]

    transcript = Transcript(
        text=redacted_text,
        language=gemini_result.get("language") or envelope.language_hint,
        confidence=float(gemini_result.get("confidence", 0.8) or 0.8),
        segments=segments,
    )

    sentiment = Sentiment(
        valence=float(gemini_result.get("overall_valence", 0.0) or 0.0),
        arousal=float(gemini_result.get("overall_arousal", 0.0) or 0.0),
    )

    markers = extract_markers(redacted_text)

    return EvidenceBundle(
        case_id=envelope.case_id,
        source_input_ids=[envelope.input_id],
        transcript=transcript,
        markers=markers,
        sentiment=sentiment,
        voice_features=None,  # populated separately by the /analyze-voice prosody engine, not here
        pii_redacted=True,
    )


def _fuse_text(envelope: InputEnvelope) -> EvidenceBundle:
    raw_text = envelope.text.body
    redacted_text, _ = redact_pii(raw_text)
    transcript = Transcript(text=redacted_text, language=envelope.language_hint, confidence=1.0, segments=[])
    markers = extract_markers(redacted_text)
    return EvidenceBundle(
        case_id=envelope.case_id,
        source_input_ids=[envelope.input_id],
        transcript=transcript,
        markers=markers,
        sentiment=None,  # no audio tone available on chat/portal
        voice_features=None,
        pii_redacted=True,
    )


# ---------------------------------------------------------------------------
# Gemini Audio Understanding call
# ---------------------------------------------------------------------------

_TRANSCRIPTION_PROMPT = """
Process this caller audio for a crisis-helpline triage system.

Requirements:
1. Produce a verbatim transcript, preserving Hindi / English / Hinglish
   code-switching exactly as spoken -- do not translate or clean it up.
2. Split into segments with start/end timestamps in seconds and, if
   more than one speaker is audible, a speaker label.
3. Detect the dominant language per segment (e.g. "hi", "en", "hi-en").
4. For each segment, classify the speaker's emotional tone as one of:
   neutral, calm, sad, fearful, angry, distressed.
5. Give one overall emotion label for the whole clip, plus an overall
   valence estimate (-1.0 very negative .. 1.0 very positive) and an
   overall arousal estimate (0.0 calm .. 1.0 highly agitated).

Do not judge risk or danger and do not make a safety recommendation --
only describe what is said and how it sounds. Risk assessment happens
in a separate step outside this task.
"""

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "full_text": {"type": "string"},
        "language": {"type": "string"},
        "confidence": {"type": "number"},
        "overall_emotion": {
            "type": "string",
            "enum": ["neutral", "calm", "sad", "fearful", "angry", "distressed"],
        },
        "overall_valence": {"type": "number"},
        "overall_arousal": {"type": "number"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string"},
                    "start_seconds": {"type": "number"},
                    "end_seconds": {"type": "number"},
                    "content": {"type": "string"},
                    "language": {"type": "string"},
                    "emotion": {
                        "type": "string",
                        "enum": ["neutral", "calm", "sad", "fearful", "angry", "distressed"],
                    },
                },
                "required": ["content"],
            },
        },
    },
    "required": ["full_text", "segments"],
}


def _call_gemini_audio(storage_uri: str, mime_type: Optional[str]) -> dict[str, Any]:
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env "
            "(backend/orchestration/config.py already reads it)."
        )

    from google import genai  # lazy import: text-only fusion never needs this SDK loaded

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    audio_input = _resolve_audio_input(client, storage_uri, mime_type)

    interaction = client.interactions.create(
        model=config.GEMINI_MODEL,
        input=[audio_input, {"type": "text", "text": _TRANSCRIPTION_PROMPT}],
        response_format=_RESPONSE_SCHEMA,
    )

    return _parse_gemini_output(interaction.output_text)


def _resolve_audio_input(client: Any, storage_uri: str, mime_type: Optional[str]) -> dict[str, Any]:
    """
    storage_uri today is expected to be a local file path (this is the
    hackathon/dev prototype -- Layer 0's own docstring calls it "a
    buffer handle" and leaves the real storage backend undecided). If
    it resolves to a real file, upload it via the Files API. If it's
    already a remote URI Gemini can fetch (gs://... or a public https
    URL), pass it straight through.

    When Twilio/Exotel + real blob storage replace this prototype, this
    is the one function that needs to change -- everything downstream
    only cares that it gets back {"type": "audio", "uri": ..., "mime_type": ...}.
    """
    path = Path(storage_uri)
    if path.exists() and path.is_file():
        uploaded = client.files.upload(file=str(path))
        return {
            "type": "audio",
            "uri": uploaded.uri,
            "mime_type": getattr(uploaded, "mime_type", None) or mime_type or "audio/wav",
        }
    if storage_uri.startswith(("http://", "https://", "gs://")):
        return {"type": "audio", "uri": storage_uri, "mime_type": mime_type or "audio/wav"}
    raise FileNotFoundError(
        f"AudioReference.storage_uri '{storage_uri}' is neither a local file "
        "nor a gs://... / http(s)://... URI Gemini can fetch."
    )


def _parse_gemini_output(output_text: str) -> dict[str, Any]:
    cleaned = (output_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# PII redaction (fallback -- replace with the real Presidio/spaCy pipeline)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, "re.Pattern[str]"]] = [
    ("PHONE", re.compile(r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("PIN_CODE", re.compile(r"\b\d{6}\b")),
]


def redact_pii(text: str) -> tuple[str, bool]:
    """
    Regex-based PII redaction fallback.

    Deliberately simple: catches phone/Aadhaar-shaped/email/PIN-code
    patterns only. It does NOT catch names, street addresses, or other
    free-text PII the way the Presidio + spaCy NER pipeline (retained
    from Detox.ai) does -- that's a heavier dependency (spaCy model
    download) I didn't want to silently pull in here. Swap this
    function's body for a call into that real pipeline before this
    touches anything beyond a demo; the call sites elsewhere in this
    file don't need to change, they just need `(redacted_text, ran_ok)`
    back.
    """
    if not text:
        return text, True
    redacted = text
    for label, pattern in _PII_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{label}]", redacted)
    return redacted, True


# ---------------------------------------------------------------------------
# Explicit danger/threat marker extraction (rule-based, NOT tone-based)
# ---------------------------------------------------------------------------

# Deliberately narrow, high-precision patterns. A marker fired here is
# treated as strong evidence downstream (see evidence_merge.py: a marker
# that fires in any one chunk is never diluted), so precision matters
# more than recall. This is a STARTING set for the SIH prototype --
# review/extend it with a PoA-Act domain SME before relying on it for
# anything beyond a demo.
_DANGER_PATTERNS: list[tuple[str, str, "re.Pattern[str]"]] = [
    (
        "immediate_danger",
        "critical",
        # Danger word and "today/tonight" word can appear in either order
        # (Hindi word order commonly puts the time word first, e.g.
        # "aaj raat maar dega") -- checked both ways within a 40-char window.
        re.compile(
            r"\b(kill|maar|marunga|marega|jaan se maar)\b.{0,40}\b(aaj|tonight|abhi|today)\b"
            r"|\b(aaj|tonight|abhi|today)\b.{0,40}\b(kill|maar|marunga|marega|jaan se maar)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit_threat",
        "high",
        re.compile(
            r"\b(will kill|going to kill|jaan se maar dunga|jaan se maar denge)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weapon_mention",
        "high",
        re.compile(r"\b(chaku|churi|knife|gun|pistol|petrol|acid)\b", re.IGNORECASE),
    ),
    (
        "abuse_disclosure",
        "medium",
        re.compile(
            r"\b(mujhe (maara|peeta)|beat(?:s|ing)? me|he hit me|she hit me)\b",
            re.IGNORECASE,
        ),
    ),
]


def extract_markers(text: str) -> list[Marker]:
    if not text:
        return []
    markers: list[Marker] = []
    for marker_type, severity, pattern in _DANGER_PATTERNS:
        for match in pattern.finditer(text):
            markers.append(
                Marker(
                    marker_type=marker_type,
                    severity=severity,
                    matched_text=match.group(0),
                    source="transcript_rule_based",
                )
            )
    return markers
