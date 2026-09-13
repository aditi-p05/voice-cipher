"""
Layer 2 (SVI / Risk) — structured, safe error handling.

Mirrors the pattern already used in services/rag/service.py
(RagValidationError(ValueError)) and services/fusion/errors.py: a small
set of stable string codes plus a message-carrying exception, so
malformed input produces a predictable, safe failure rather than a raw
traceback or a silently fabricated score.
"""

from __future__ import annotations

INVALID_INPUT = "INVALID_INPUT"


class SviError(ValueError):
    """Safe validation failure in Layer 2 (SVI / Risk)."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)
