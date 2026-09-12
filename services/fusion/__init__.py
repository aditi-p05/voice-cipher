"""
Layer 1 (Fusion / Privacy) — Member 2's module.

Public interface for other layers:

    from services.fusion import build_evidence_bundle
    bundle = build_evidence_bundle(input_envelope)

Do not import services.fusion.* submodules directly from outside this
package (per CONTRACTS.md integration rule 4: depend on schemas/adapters,
not another member's internals). `EvidenceBundle` and its nested models
are the shared schema and are fine to import for type hints.
"""

from services.fusion.evidence_bundle import (
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EvidenceBundle,
    Marker,
    Sentiment,
    Transcript,
    TranscriptSegment,
    VoiceFeatures,
)
from services.fusion.errors import (
    INVALID_INPUT,
    MODEL_FAILED,
    STT_FAILED,
    TIMEOUT,
    UNSUPPORTED_LANGUAGE,
    FusionError,
)
from services.fusion.fusion_service import build_evidence_bundle

__all__ = [
    "build_evidence_bundle",
    "EvidenceBundle",
    "Transcript",
    "TranscriptSegment",
    "Marker",
    "Sentiment",
    "VoiceFeatures",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "FusionError",
    "UNSUPPORTED_LANGUAGE",
    "STT_FAILED",
    "MODEL_FAILED",
    "TIMEOUT",
    "INVALID_INPUT",
]
