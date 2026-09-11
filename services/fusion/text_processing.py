"""
Layer 1 (Fusion / Privacy) — text normalization and source extraction.

Handles turning an InputEnvelope's declared text-bearing fields (chatbot
message, portal free text, portal structured fields) into a single
normalized string for downstream language detection / redaction /
marker / sentiment analysis. Performs no redaction and no analysis
itself — this module only assembles and cleans raw text.
"""

from __future__ import annotations

import re
import unicodedata

from backend.ingestion.input_envelope import InputEnvelope
from backend.models.enums import Modality


def normalize_text(text: str) -> str:
    """
    Safe, conservative normalization: Unicode NFC normalization, collapse
    internal whitespace/newlines to single spaces, strip leading/trailing
    whitespace. Deliberately does not lowercase or strip punctuation —
    that is left to individual analyzers that need it.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _structured_data_as_text(envelope: InputEnvelope) -> str:
    """
    Render portal structured fields as a short, explainable text blob so
    they can flow through the same normalize/redact/marker pipeline as
    free text, per responsibility 1 ("process structured portal fields
    where appropriate"). Category/location are treated as low-sensitivity
    context; `fields` are treated the same as free text for redaction
    purposes since callers can put anything in them.
    """
    if envelope.structured_data is None:
        return ""

    parts: list[str] = []
    if envelope.structured_data.category:
        parts.append(f"category: {envelope.structured_data.category}")
    if envelope.structured_data.location:
        parts.append(f"location: {envelope.structured_data.location}")
    for key, value in envelope.structured_data.fields.items():
        parts.append(f"{key}: {value}")

    return normalize_text(". ".join(parts))


def extract_source_text(envelope: InputEnvelope) -> str:
    """
    Combine every text-bearing source declared in `envelope.modalities`
    into one normalized string. Does not touch audio — STT output is
    combined separately in fusion_service.py because it requires the
    pluggable SpeechToTextProvider, not just envelope inspection.
    """
    pieces: list[str] = []

    if Modality.TEXT in envelope.modalities and envelope.text is not None:
        pieces.append(normalize_text(envelope.text.body))

    if Modality.STRUCTURED_DATA in envelope.modalities:
        structured_text = _structured_data_as_text(envelope)
        if structured_text:
            pieces.append(structured_text)

    return normalize_text(" ".join(p for p in pieces if p))
