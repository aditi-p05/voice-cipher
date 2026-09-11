"""
Layer 1 (Fusion / Privacy) — optional Whisper-backed STT provider.

Not used by default and not imported anywhere else in this package. The
`openai-whisper` / `faster-whisper` dependency and model download only
happen if this class is explicitly instantiated, so `pytest -q` and the
mock pipeline never require downloading a model.

This is currently a stub: this repository never gives Layer 1 raw audio
bytes (see mock_provider.py for why), so there is nothing for a real STT
engine to consume yet. When Layer 0 (or a storage adapter) starts
exposing actual audio bytes/streams for a `storage_uri`, fill in
`_load_model` and `_read_audio_bytes` below.
"""

from __future__ import annotations

from backend.ingestion.input_envelope import AudioReference
from services.fusion.errors import STT_FAILED, FusionError
from services.fusion.stt.base import STTResult, SpeechToTextProvider


class WhisperSTTProvider(SpeechToTextProvider):
    """Isolates the Whisper dependency so nothing else imports it directly."""

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None  # lazy-loaded on first real use

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import whisper  # type: ignore  # heavy optional dependency
        except ImportError as exc:
            raise FusionError(
                STT_FAILED,
                "Whisper STT provider requested but the 'openai-whisper' "
                "package is not installed in this environment.",
                details={"provider": "whisper"},
            ) from exc
        self._model = whisper.load_model(self._model_size)
        return self._model

    def transcribe(self, audio: AudioReference) -> STTResult:
        # No raw audio bytes reach Layer 1 in the current architecture
        # (see mock_provider.py docstring). Fail safely and explicitly
        # rather than fabricating a transcript.
        raise FusionError(
            STT_FAILED,
            "WhisperSTTProvider has no audio byte source to transcribe in "
            "the current architecture.",
            details={"provider": "whisper", "audio_ref_id": audio.audio_ref_id},
        )
