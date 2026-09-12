"""
Layer 1 (Fusion / Privacy) — optional acoustic feature extraction.

Voice features are optional everywhere downstream (CONTRACTS.md 6.2:
`voice_features` may be null; "the system must still work when
voice_features = null"). Because Layer 0's `AudioReference` never carries
raw audio bytes, this extractor currently always returns `None` for real
traffic — that is the correct, honest behavior, not a bug. Tests can
inject a fixture provider to exercise the "voice features present" path
with synthetic numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.ingestion.input_envelope import AudioReference


class VoiceFeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio: AudioReference) -> dict | None:
        """
        Return a dict matching services.fusion.evidence_bundle.VoiceFeatures
        fields, or None if extraction is not possible/available.
        """
        raise NotImplementedError


class NullVoiceFeatureExtractor(VoiceFeatureExtractor):
    """Default: no raw audio bytes are available, so always return None."""

    def extract(self, audio: AudioReference) -> dict | None:
        return None


class FixtureVoiceFeatureExtractor(VoiceFeatureExtractor):
    """Test/demo-only extractor returning pre-registered synthetic features."""

    def __init__(self, fixtures: dict[str, dict] | None = None):
        self._fixtures = fixtures or {}

    def register_fixture(self, audio_ref_id: str, features: dict) -> None:
        self._fixtures[audio_ref_id] = features

    def extract(self, audio: AudioReference) -> dict | None:
        return self._fixtures.get(audio.audio_ref_id)
