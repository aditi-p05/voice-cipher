"""
Layer 4B (Orchestration) adapter for Layer 1 (Fusion / Privacy).

Calls Member 2's REAL public adapter directly — services.fusion is fully
implemented, so no mock is used or needed here. This module only adds a
safe, structured failure boundary and an optional timeout; it must never
import fusion internals (redaction/, stt/, markers/, etc. are off-limits).
"""

from __future__ import annotations

from typing import Callable, Optional

from backend.ingestion.input_envelope import InputEnvelope
from backend.orchestration.errors import stage_error
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.errors import FusionError
from services.fusion.evidence_bundle import EvidenceBundle
from services.fusion.fusion_service import build_evidence_bundle

FusionCallable = Callable[[InputEnvelope], EvidenceBundle]


def run_fusion(
    envelope: InputEnvelope,
    *,
    fusion_fn: Optional[FusionCallable] = None,
    timeout_seconds: float = 10.0,
) -> tuple[Optional[EvidenceBundle], Optional[dict]]:
    """
    Returns (evidence_bundle, error). Exactly one is non-None.

    `fusion_fn` defaults to the real build_evidence_bundle; tests may
    inject a fake/failing callable to exercise FUSION_FAILED / TIMEOUT
    without touching the real fusion implementation.
    """
    active_fn = fusion_fn or build_evidence_bundle
    try:
        bundle = run_with_timeout(lambda: active_fn(envelope), seconds=timeout_seconds)
        return bundle, None
    except NodeTimeoutError:
        return None, stage_error("fusion", "Evidence fusion timed out.", code="TIMEOUT")
    except FusionError as exc:
        return None, stage_error("fusion", exc.message)
    except Exception:
        # Final safety net: never leak a raw provider exception upward.
        return None, stage_error("fusion", "Evidence fusion failed unexpectedly.")
