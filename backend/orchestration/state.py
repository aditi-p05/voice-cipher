"""
Layer 4B (Orchestration) — pipeline state.

A TypedDict (LangGraph's native state shape) holding CANONICAL shared
objects -- InputEnvelope, EvidenceBundle -- plus contract-shaped dicts
for SVIResult / ServiceRecommendation / SupportDecision.

`errors` uses an explicit LangGraph reducer (`operator.add`) rather than
the default overwrite-on-write behavior. This matters because SERVICE
and SUPPORT run as true parallel branches off SVI (see graph.py): if two
nodes in the same superstep both wrote a full "errors" list, the default
reducer would silently keep only one branch's list and drop the other's
error. With `operator.add`, each node returns ONLY the error(s) it itself
produced (see nodes.py) and LangGraph concatenates them -- so a support
failure and a rag failure in the same run are both preserved, never one
overwriting the other.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from backend.ingestion.input_envelope import InputEnvelope
from services.fusion.evidence_bundle import EvidenceBundle


class PipelineState(TypedDict, total=False):
    input_envelope: Optional[InputEnvelope]
    evidence_bundle: Optional[EvidenceBundle]
    svi_result: Optional[dict[str, Any]]
    service_recommendation: Optional[dict[str, Any]]
    support_decision: Optional[dict[str, Any]]
    errors: Annotated[list[dict[str, Any]], operator.add]


def initial_state(envelope: Optional[InputEnvelope]) -> PipelineState:
    return PipelineState(
        input_envelope=envelope,
        evidence_bundle=None,
        svi_result=None,
        service_recommendation=None,
        support_decision=None,
        errors=[],
    )

