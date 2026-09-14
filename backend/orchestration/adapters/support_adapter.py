"""
Layer 4B (Orchestration) adapter for the Support Engine (services/support).

A support failure must NEVER block or delay the service/escalation path.
This adapter guarantees that: any exception becomes a structured
SUPPORT_FAILED error, and the caller (graph.py) treats "support
unavailable" exactly like "support disabled" for safety purposes.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from backend.orchestration.errors import stage_error
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.support.errors import SupportError
from services.support.support_service import decide_support

SupportCallable = Callable[[Any, Any], dict]


def run_support(
    evidence_bundle: Any,
    svi_result: Any,
    *,
    support_fn: Optional[SupportCallable] = None,
    timeout_seconds: float = 5.0,
) -> tuple[Optional[dict], Optional[dict]]:
    active_fn = support_fn or decide_support
    try:
        result = run_with_timeout(
            lambda: active_fn(evidence_bundle, svi_result), seconds=timeout_seconds
        )
        return result, None
    except NodeTimeoutError:
        return None, stage_error("support", "Support decision timed out.", code="TIMEOUT")
    except SupportError as exc:
        return None, stage_error("support", exc.message)
    except Exception:
        return None, stage_error("support", "Support decision failed unexpectedly.")
