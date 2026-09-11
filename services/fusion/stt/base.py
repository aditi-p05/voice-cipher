"""
Layer 1 (Fusion / Privacy) — pluggable speech-to-text abstraction.

Nothing in this module or its callers should hard-code a single STT
vendor. `SpeechToTextProvider` is the seam; concrete providers (mock,
Whisper, a cloud API, ...) live behind it and are swapped via
fusion_service configuration, never by changing call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.ingestion.input_envelope import AudioReference


@dataclass(frozen=True)
class STTSegment:
    text: str
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True)
class STTResult:
    """Raw (un-redacted) transcription output. Never persisted as-is."""

    text: str
    language: str | None
    confidence: float
    segments: list[STTSegment] = field(default_factory=list)


class SpeechToTextProvider(ABC):
    """Common interface every STT backend must implement."""

    @abstractmethod
    def transcribe(self, audio: AudioReference) -> STTResult:
        """
        Transcribe the audio referenced by `audio`.

        Implementations must raise `services.fusion.errors.FusionError`
        with code STT_FAILED (or TIMEOUT) on failure — never leak a raw
        provider exception to the caller.
        """
        raise NotImplementedError
