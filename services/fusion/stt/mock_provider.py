"""
Layer 1 (Fusion / Privacy) — mock speech-to-text provider.

Why this exists (read before "fixing" it):
Layer 0's `AudioReference` deliberately never carries raw audio bytes
(see backend/ingestion/input_envelope.py) — only a reference/metadata.
That means there is currently no real audio signal anywhere in this
repository for any STT engine, real or otherwise, to transcribe.

This provider is therefore the *only* correct provider for the current
system: it never fabricates speech content it has no basis for. It
returns a clearly-flagged "unavailable" result (confidence 0.0, no text)
unless a caller has explicitly registered a synthetic fixture transcript
for a given `audio_ref_id` — which is how tests simulate "voice with a
transcript" using synthetic data only, without pretending a mock is a
genuine transcription of real speech.

A real provider (e.g. Whisper) can be added later behind
`SpeechToTextProvider` without touching fusion_service.py.
"""

from __future__ import annotations

from backend.ingestion.input_envelope import AudioReference
from services.fusion.stt.base import STTResult, SpeechToTextProvider

UNAVAILABLE_TEXT = ""


class MockSTTProvider(SpeechToTextProvider):
    """
    Deterministic, no-download STT stand-in.

    `fixtures` maps `audio_ref_id -> STTResult` for synthetic test/demo
    data. Any audio reference not present in `fixtures` is treated as
    "no real transcript available" rather than being fabricated.
    """

    def __init__(self, fixtures: dict[str, STTResult] | None = None):
        self._fixtures = fixtures or {}

    def register_fixture(self, audio_ref_id: str, result: STTResult) -> None:
        self._fixtures[audio_ref_id] = result

    def transcribe(self, audio: AudioReference) -> STTResult:
        fixture = self._fixtures.get(audio.audio_ref_id)
        if fixture is not None:
            return fixture

        return STTResult(
            text=UNAVAILABLE_TEXT,
            language=None,
            confidence=0.0,
            segments=[],
        )
