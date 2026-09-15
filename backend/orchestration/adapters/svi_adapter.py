"""Layer 2 — deterministic Stress Vulnerability Index adapter.

The local ``services.svi.calculate_svi`` implementation is the only
default scoring path. Alternative providers must be explicitly injected
through ``Providers(svi_provider=...)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from backend.orchestration import config, errors
from backend.orchestration.timeout import NodeTimeoutError, run_with_timeout
from services.fusion.evidence_bundle import EvidenceBundle
from services.svi import calculate_svi

SVIProvider = Callable[[EvidenceBundle], dict]


def run_svi(
    evidence_bundle: EvidenceBundle,
    *,
    provider: Optional[SVIProvider] = None,
    timeout_seconds: Optional[float] = None,
) -> tuple[Optional[dict], Optional[dict]]:
    budget = timeout_seconds if timeout_seconds is not None else config.SVI_TIMEOUT_SECONDS

    def score() -> dict:
        # The deterministic Layer 2 service is the safe local default. It
        # needs no network/API key and returns the contract expected by RAG
        # and Support. Callable and `.calculate()` providers remain
        # injectable for tests and future model-backed deployments.
        if provider is None:
            result: Any = calculate_svi(evidence_bundle)
        elif callable(provider):
            result = provider(evidence_bundle)
        elif callable(getattr(provider, "calculate", None)):
            result = provider.calculate(evidence_bundle)
        else:
            raise TypeError("SVI provider must be callable or expose calculate()")
        dump = getattr(result, "model_dump", None)
        return dump(mode="json") if callable(dump) else result

    try:
        return run_with_timeout(score, seconds=budget), None
    except NodeTimeoutError as exc:
        return None, errors.stage_error("svi", str(exc), code=errors.TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        return None, errors.stage_error("svi", f"SVI failed: {exc}")
