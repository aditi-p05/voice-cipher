"""
Layer 4B (Orchestration) — configuration.

Timeouts are read from the environment so operators can tune them per
deployment (a slower real STT/RAG provider in production vs. instant
mock providers in tests/demo) without a code change or redeploy of logic.
Defaults match what was previously hardcoded in Providers.
"""

from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


FUSION_TIMEOUT_SECONDS = _float_env("NHAA_FUSION_TIMEOUT_SECONDS", 10.0)
RAG_TIMEOUT_SECONDS = _float_env("NHAA_RAG_TIMEOUT_SECONDS", 10.0)
SUPPORT_TIMEOUT_SECONDS = _float_env("NHAA_SUPPORT_TIMEOUT_SECONDS", 5.0)
