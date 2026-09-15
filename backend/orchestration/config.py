"""
Layer 4B (Orchestration) — configuration.

Timeouts are read from the environment so operators can tune them per
deployment (a slower real STT/RAG provider in production vs. instant
mock providers in tests/demo) without a code change or redeploy of logic.
Defaults match what was previously hardcoded in Providers.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


# Load local developer credentials when the API starts. Existing environment
# variables always take precedence, which keeps container/CI deployments safe.
load_dotenv(override=False)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


FUSION_TIMEOUT_SECONDS = _float_env("NHAA_FUSION_TIMEOUT_SECONDS", 10.0)
SVI_TIMEOUT_SECONDS = _float_env("NHAA_SVI_TIMEOUT_SECONDS", 15.0)
RAG_TIMEOUT_SECONDS = _float_env("NHAA_RAG_TIMEOUT_SECONDS", 10.0)
SUPPORT_TIMEOUT_SECONDS = _float_env("NHAA_SUPPORT_TIMEOUT_SECONDS", 5.0)
# SVI is now a network-bound Gemini call like the others, so it needs its own
# budget. Previously SVI was pure local arithmetic and had no timeout at all.
SVI_TIMEOUT_SECONDS = _float_env("NHAA_SVI_TIMEOUT_SECONDS", 10.0)

# Now actually used by fusion_adapter.py, svi_adapter.py, rag_adapter.py and
# support_adapter.py's default (Gemini-backed) providers.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
