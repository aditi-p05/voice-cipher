"""
Layer 4B (Support Engine) — data shapes.

`SupportDecision` is NOT one of the CONTRACTS.md shared schemas (it isn't
consumed by Members 2-4). It is Layer 4B's own output, handed to
orchestration and, from there, to the dashboard (Member 6) alongside
`ServiceRecommendation`. If the team later wants this formalized as a
numbered CONTRACTS.md schema, propose it there rather than treating this
module as authoritative on its own.

Hard rule: nothing here is medical treatment, therapy, or a clinical
intervention. `disclaimer` must always accompany an enabled decision.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

SUPPORT_DECISION_SCHEMA_VERSION = "1.0.0"

NON_CLINICAL_DISCLAIMER = (
    "This is an optional, non-clinical self-regulation aid. It is not "
    "medical treatment, therapy, or a diagnosis, and it never replaces "
    "human/professional support."
)


class SupportPreset(BaseModel):
    """A curated, non-generative support suggestion (audio/guidance lookup)."""

    preset_id: str
    label: str
    modality: str  # e.g. "guided_breathing", "curated_audio", "grounding_text"
    description: str

    model_config = ConfigDict(extra="forbid")


class SupportDecision(BaseModel):
    """
    Layer 4B's risk-gated support decision.

    `support_enabled` and `mode` are the fields orchestration/dashboard must
    check before ever surfacing a preset to a user. A disabled/restricted
    decision must never be paired with a preset.
    """

    schema_version: str = SUPPORT_DECISION_SCHEMA_VERSION
    case_id: str
    risk_tier: str
    mode: str  # "enabled" | "restricted" | "disabled"
    support_enabled: bool
    preset: Optional[SupportPreset] = None
    reason: str
    disclaimer: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(extra="forbid")
