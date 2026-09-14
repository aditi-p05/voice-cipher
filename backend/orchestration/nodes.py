"""
Layer 4B (Orchestration) — graph node functions.

Each node is a plain function of (state, providers) -> PARTIAL state
update (only the keys it changed). This is deliberate, not just style:
`errors` uses an accumulating reducer (see state.py) that concatenates
whatever each node returns onto the existing list. A node that returned
the *full* history (`existing + new`) would double-count on every write;
returning only its own new error(s) is what makes the reducer correct.

Nodes are independent of LangGraph's runtime so they can be unit-tested
directly, without building/compiling a graph at all. `graph.py` is the
only module that wires these into an actual StateGraph.

Failure philosophy (Developer 5 task brief):
  - Fusion fails            -> no evidence; SVI/RAG/Support all skipped;
                                 do not fabricate evidence.
  - SVI fails               -> no score; RAG/Support skipped (both need
                                 risk_tier); do not fabricate a score.
  - RAG fails               -> structured failure only; Support and the
                                 rest of the pipeline continue.
  - Support fails           -> structured failure only; must NEVER affect
                                 the service/critical path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from backend.orchestration import config
from backend.orchestration.adapters.fusion_adapter import run_fusion
from backend.orchestration.adapters.rag_adapter import run_rag
from backend.orchestration.adapters.support_adapter import run_support
from backend.orchestration.adapters.svi_adapter import SVIProvider, run_svi
from backend.orchestration.state import PipelineState


@dataclass
class Providers:
    """Dependency-injection bundle. Every field has a safe real default;
    tests override individual fields to simulate failure/timeout without
    touching real fusion/rag/support/svi implementations.

    Timeout defaults come from backend.orchestration.config (environment-
    configurable) rather than being hardcoded here, so an operator can
    tune them per deployment without touching this module."""

    fusion_fn: Optional[Callable[[Any], Any]] = None
    svi_provider: Optional[SVIProvider] = None
    rag_fn: Optional[Callable[[Any, Any], dict]] = None
    support_fn: Optional[Callable[[Any, Any], dict]] = None
    fusion_timeout_seconds: float = field(default_factory=lambda: config.FUSION_TIMEOUT_SECONDS)
    rag_timeout_seconds: float = field(default_factory=lambda: config.RAG_TIMEOUT_SECONDS)
    support_timeout_seconds: float = field(default_factory=lambda: config.SUPPORT_TIMEOUT_SECONDS)


def fusion_node(state: PipelineState, providers: Providers) -> dict:
    envelope = state.get("input_envelope")
    if envelope is None:
        # Entry point for the post-fusion graph (see graph.py /
        # run_pipeline_from_evidence): evidence was already produced
        # upstream (e.g. merged across voice chunks). Nothing to do.
        return {}

    bundle, error = run_fusion(
        envelope,
        fusion_fn=providers.fusion_fn,
        timeout_seconds=providers.fusion_timeout_seconds,
    )
    return {"evidence_bundle": bundle, "errors": [error] if error else []}


def svi_node(state: PipelineState, providers: Providers) -> dict:
    if state.get("evidence_bundle") is None:
        # Fusion already failed upstream; do not attempt SVI on nothing.
        return {}

    result, error = run_svi(state["evidence_bundle"], provider=providers.svi_provider)
    update: dict = {"errors": [error] if error else []}
    if result is not None:
        update["svi_result"] = result
    return update


def service_node(state: PipelineState, providers: Providers) -> dict:
    if state.get("evidence_bundle") is None or state.get("svi_result") is None:
        # Nothing to recommend against; upstream already failed and was recorded.
        return {}

    result, error = run_rag(
        state["evidence_bundle"],
        state["svi_result"],
        rag_fn=providers.rag_fn,
        timeout_seconds=providers.rag_timeout_seconds,
    )
    update: dict = {"errors": [error] if error else []}
    if result is not None:
        update["service_recommendation"] = result
    return update


def support_node(state: PipelineState, providers: Providers) -> dict:
    if state.get("evidence_bundle") is None or state.get("svi_result") is None:
        return {}

    result, error = run_support(
        state["evidence_bundle"],
        state["svi_result"],
        support_fn=providers.support_fn,
        timeout_seconds=providers.support_timeout_seconds,
    )
    update: dict = {"errors": [error] if error else []}
    if result is not None:
        update["support_decision"] = result
    return update


def dashboard_node(state: PipelineState, providers: Providers) -> dict:
    """
    Terminal join node. SERVICE and SUPPORT run as true parallel branches
    (see graph.py) and both feed into this node; LangGraph only invokes it
    once both predecessors have completed, so no `service_recommendation`/
    `support_decision` race is possible here. Does not build the React
    dashboard (Member 6's job) -- it only marks the graph run's endpoint.
    See run.py's `build_dashboard_case_view` for the exact handoff shape.
    """
    return {}
