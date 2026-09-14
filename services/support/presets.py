"""
Layer 4B (Support Engine) — curated preset lookup.

Deliberately a static lookup table, not a generative or ML system (see
Developer 5 task brief: "keep this lightweight ... a lookup table is
sufficient"). Legacy Detox.ai raga mappings are usable only as a *source
of curated audio ideas*, never as a claim that a preset treats trauma,
depression, or any clinical condition.

Buckets are coarse on purpose: sentiment/arousal are dimensional signals
from Layer 1, not precise measurements, so fine-grained thresholds would
imply false precision.
"""

from __future__ import annotations

from services.support.models import SupportPreset

# valence: -1 (negative) .. 1 (positive); arousal: 0 (calm) .. 1 (activated)
_PRESETS: dict[str, SupportPreset] = {
    "grounding_calm": SupportPreset(
        preset_id="grounding_calm",
        label="Grounding / slow breathing",
        modality="guided_breathing",
        description="A brief guided slow-breathing exercise for acute fear or activation.",
    ),
    "gentle_regulation": SupportPreset(
        preset_id="gentle_regulation",
        label="Gentle regulation audio",
        modality="curated_audio",
        description="Calm, low-tempo curated audio for withdrawal/low-mood indicators.",
    ),
    "social_support_guidance": SupportPreset(
        preset_id="social_support_guidance",
        label="Social connection guidance",
        modality="grounding_text",
        description="Short guidance on reconnecting with a trusted support person.",
    ),
    "neutral_checkin": SupportPreset(
        preset_id="neutral_checkin",
        label="Simple check-in",
        modality="grounding_text",
        description="A brief, low-intensity check-in prompt for neutral/low-signal cases.",
    ),
}


def select_preset(valence: float | None, arousal: float | None) -> SupportPreset:
    """
    Map dimensional sentiment to one curated preset. Never raises: missing
    sentiment falls back to the neutral check-in rather than guessing.
    """
    if valence is None or arousal is None:
        return _PRESETS["neutral_checkin"]

    if arousal >= 0.6 and valence < 0:
        return _PRESETS["grounding_calm"]
    if valence < -0.2 and arousal < 0.6:
        return _PRESETS["gentle_regulation"]
    if valence < 0:
        return _PRESETS["social_support_guidance"]
    return _PRESETS["neutral_checkin"]
