"""
Layer 4B (Orchestration) adapter for Layer 4A (Service / RAG).

Calls Member 4's REAL public adapter directly -- services.rag is fully
implemented. A RAG failure must never block the pipeline: it becomes a
structured RAG_FAILED error, and the dashboard/operator is told
procedural guidance was unavailable rather than seeing nothing or a
fabricated recommendation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from backend.orchestration.errors import stage_error
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.rag.service import RagUnavailableError, RagValidationError, generate_recommendation

RagCallable = Callable[[Any, Any], dict]


def run_rag(
    evidence_bundle: Any,
    svi_result: Any,
    *,
    rag_fn: Optional[RagCallable] = None,
    timeout_seconds: float = 10.0,
) -> tuple[Optional[dict], Optional[dict]]:
    active_fn = rag_fn or generate_recommendation
    try:
        result = run_with_timeout(
            lambda: active_fn(evidence_bundle, svi_result), seconds=timeout_seconds
        )
        return result, None
    except NodeTimeoutError:
        return None, stage_error("rag", "Procedural guidance retrieval timed out.", code="TIMEOUT")
    except RagValidationError as exc:
        return None, stage_error("rag", str(exc))
    except RagUnavailableError:
        return None, stage_error("rag", "Procedural guidance was unavailable.")
    except Exception:
        return None, stage_error("rag", "Procedural guidance was unavailable.")
