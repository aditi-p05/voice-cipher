"""
backend/routing/modality_router.py

*** STUB — this did NOT ship in the zip you gave me. ***

`backend/ingestion/intake_service.py` already does
`from backend.routing.modality_router import RoutingResult, route`
unconditionally -- without this module, nothing in Layer 0 (and
therefore nothing behind it: cases, orchestration, the whole FastAPI
app) can even be imported.

This is Layer 0 / Member 1's territory (their own docstrings elsewhere
describe modality routing as part of the input layer, separate from
validation and dispatch). Today `RoutingResult` is only ever *stored*
on `IntakeOutcome` -- nothing else in this codebase branches on it --
so this stub picks a reasonable, low-risk contract: decide which
downstream queue an envelope belongs to, deterministically, from its
channel and modalities. Replace with Member 1's real routing rules
(e.g. priority queues, per-modality fan-out) when available; nothing
downstream should need to change since nothing downstream reads
RoutingResult today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.ingestion.input_envelope import InputEnvelope
from backend.models.enums import Channel, Modality


@dataclass
class RoutingResult:
    queue_name: str
    reasons: list[str] = field(default_factory=list)


_CHANNEL_QUEUES: dict[Channel, str] = {
    Channel.VOICE_CALL: "voice-intake",
    Channel.CHATBOT: "chat-intake",
    Channel.PORTAL: "portal-intake",
}


def route(envelope: InputEnvelope) -> RoutingResult:
    """Deterministic channel -> queue mapping. No risk/ML logic here,
    matching Layer 0's own rule that this layer stays evidence-blind."""
    queue = _CHANNEL_QUEUES.get(envelope.channel, "default-intake")
    reasons = [f"channel={envelope.channel.value}"]
    if Modality.AUDIO in envelope.modalities:
        reasons.append("has_audio")
    if Modality.TEXT in envelope.modalities:
        reasons.append("has_text")
    if Modality.STRUCTURED_DATA in envelope.modalities:
        reasons.append("has_structured_data")
    return RoutingResult(queue_name=queue, reasons=reasons)
