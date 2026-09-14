"""
Layer 4B (Orchestration) — LangGraph state machine.

    INGEST (already produced by Layer 0) -> FUSION -> SVI -> RISK_TIER
        -> {SERVICE, SUPPORT} -> DASHBOARD

SERVICE and SUPPORT run as TRUE parallel branches off SVI, matching the
task brief's graph concept. This is safe here (and wasn't in an earlier
version of this module) because `PipelineState.errors` now uses an
explicit `operator.add` reducer (see state.py) instead of default
overwrite-on-write -- so a rag failure recorded by SERVICE and a support
failure recorded by SUPPORT in the same run are both kept, not one
silently dropping the other. `service_recommendation` and
`support_decision` never conflict since each branch only ever writes its
own key.

"RISK_TIER" is not a separate node: risk_tier lives inside svi_result
(CONTRACTS.md 6.3) and is read directly by service_node/support_node.

Two compiled graphs are exposed:
  - build_graph(...)            entry point FUSION -- the normal path for
                                 a single InputEnvelope (chat/portal/a
                                 single voice interaction).
  - build_post_fusion_graph(...) entry point SVI -- for callers (e.g.
                                 voice_session.py) that already produced
                                 an EvidenceBundle themselves, such as one
                                 merged across several voice chunks, and
                                 must not re-run fusion on it.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from backend.orchestration.nodes import (
    Providers,
    dashboard_node,
    fusion_node,
    service_node,
    support_node,
    svi_node,
)
from backend.orchestration.state import PipelineState


def _add_shared_nodes(graph: StateGraph, providers: Providers) -> None:
    graph.add_node("svi", partial(svi_node, providers=providers))
    graph.add_node("service", partial(service_node, providers=providers))
    graph.add_node("support", partial(support_node, providers=providers))
    graph.add_node("dashboard", partial(dashboard_node, providers=providers))

    graph.add_edge("svi", "service")
    graph.add_edge("svi", "support")
    graph.add_edge("service", "dashboard")
    graph.add_edge("support", "dashboard")
    graph.add_edge("dashboard", END)


def build_graph(providers: Providers | None = None):
    """Compile the full pipeline, starting from a raw InputEnvelope."""
    active_providers = providers or Providers()
    graph = StateGraph(PipelineState)

    graph.add_node("fusion", partial(fusion_node, providers=active_providers))
    _add_shared_nodes(graph, active_providers)

    graph.set_entry_point("fusion")
    graph.add_edge("fusion", "svi")

    return graph.compile()


def build_post_fusion_graph(providers: Providers | None = None):
    """
    Compile the pipeline starting from SVI, for callers that already have
    an EvidenceBundle (e.g. merged across multiple voice chunks) and must
    not re-run fusion on it.
    """
    active_providers = providers or Providers()
    graph = StateGraph(PipelineState)

    _add_shared_nodes(graph, active_providers)
    graph.set_entry_point("svi")

    return graph.compile()
