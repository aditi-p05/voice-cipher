"""
Layer 0 — deterministic modality router.

Sole responsibility: "Given this intake, what evidence/modalities are
available?" It NEVER makes vulnerability, risk, or escalation decisions —
those belong to later layers. Output is a plain structured result that
Layer 1 consumes to decide which analyzers to run.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.models.enums import Modality
from backend.ingestion.input_envelope import InputEnvelope


@dataclass(frozen=True)
class RoutingResult:
    case_id: str
    input_id: str
    available_modalities: list[Modality]
    has_audio: bool
    has_text: bool
    has_structured_data: bool
    # Signals to Layer 1 that a transcript is expected to be produced from
    # audio (via STT), without Layer 0 performing or assuming anything
    # about that transcript's content.
    transcript_expected: bool


def route(envelope: InputEnvelope) -> RoutingResult:
    modalities = envelope.modalities
    has_audio = Modality.AUDIO in modalities
    has_text = Modality.TEXT in modalities
    has_structured = Modality.STRUCTURED_DATA in modalities

    return RoutingResult(
        case_id=envelope.case_id,
        input_id=envelope.input_id,
        available_modalities=list(modalities),
        has_audio=has_audio,
        has_text=has_text,
        has_structured_data=has_structured,
        transcript_expected=has_audio,
    )
