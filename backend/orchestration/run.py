"""
Layer 4B (Orchestration) — public entrypoints.

    run_pipeline(envelope) -> dashboard case dict
    run_pipeline_from_evidence(evidence_bundle) -> dashboard case dict

`run_pipeline` is what Member 6 (or anything else) calls for a normal,
single-interaction input (chat message, portal complaint, one voice
interaction). `run_pipeline_from_evidence` is for callers that already
produced an EvidenceBundle themselves -- currently only
`voice_session.py`, which merges evidence across repeated /voice/chunk
calls before scoring -- and must not re-run fusion on it.

`providers` is exposed on both for tests and for swapping in Member 3's
real SVI implementation once it exists, without touching graph internals.
"""

from __future__ import annotations

from typing import Any, Optional

from backend.ingestion.input_envelope import InputEnvelope
from backend.orchestration.graph import build_graph, build_post_fusion_graph
from backend.orchestration.nodes import Providers
from backend.orchestration.state import initial_state
from services.fusion.evidence_bundle import EvidenceBundle


def run_pipeline(envelope: InputEnvelope, *, providers: Providers | None = None) -> dict[str, Any]:
    graph = build_graph(providers)
    final_state = graph.invoke(initial_state(envelope))
    return build_dashboard_case_view(final_state)


def run_pipeline_from_evidence(
    evidence_bundle: EvidenceBundle,
    *,
    providers: Providers | None = None,
) -> dict[str, Any]:
    graph = build_post_fusion_graph(providers)
    state = initial_state(envelope=None)
    state["evidence_bundle"] = evidence_bundle
    final_state = graph.invoke(state)
    return build_dashboard_case_view(final_state)


def build_dashboard_case_view(state: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate the pipeline's final state into the handoff shape for
    Member 6's dashboard/API layer.

    NOTE: this is NOT a numbered CONTRACTS.md schema yet -- it is Layer
    4B's proposed handoff shape (see docs/proposals/dashboard_case_view.md
    for the formal proposal). If Member 6 needs a different/stricter
    shape, that proposal is the place to negotiate it, not a silent
    dependency on this function's current output.

    `input_envelope` may be None (the post-fusion entrypoint has no
    single canonical envelope -- e.g. a voice session merged from several
    chunks); in that case case/input identity falls back to the merged
    EvidenceBundle's own fields.
    """
    envelope: Optional[InputEnvelope] = state.get("input_envelope")
    evidence_bundle = state.get("evidence_bundle")
    svi_result = state.get("svi_result")
    service_recommendation = state.get("service_recommendation")

    if envelope is not None:
        case_id = envelope.case_id
        input_id: Optional[str] = envelope.input_id
        channel = envelope.channel.value if hasattr(envelope.channel, "value") else envelope.channel
    else:
        case_id = evidence_bundle.case_id if evidence_bundle is not None else "UNKNOWN"
        input_id = (
            evidence_bundle.source_input_ids[-1]
            if evidence_bundle is not None and evidence_bundle.source_input_ids
            else None
        )
        channel = "voice_call_session"

    # Read the actual contract field rather than truthiness of the dict --
    # a malformed/unexpected-shape recommendation must never be treated as
    # "ready for operator confirmation" just because it happens to be a
    # non-empty mapping.
    requires_confirmation = bool(
        isinstance(service_recommendation, dict)
        and service_recommendation.get("requires_operator_confirmation") is True
    )

    return {
        "schema_version": "1.0.0",
        "case_id": case_id,
        "input_id": input_id,
        "channel": channel,
        "evidence_available": evidence_bundle is not None,
        "evidence_bundle": evidence_bundle.model_dump(mode="json") if evidence_bundle is not None else None,
        "svi_result": svi_result,
        "risk_tier": svi_result.get("risk_tier") if svi_result else None,
        "service_recommendation": service_recommendation,
        "support_decision": state.get("support_decision"),
        "requires_operator_confirmation": requires_confirmation,
        "errors": state.get("errors", []),
    }
