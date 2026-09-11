"""
Layer 1 (Fusion / Privacy) — structured, safe error handling.

Mirrors the pattern already established in backend/ingestion/validator.py
and backend/core/errors.py: a small typed exception with a stable machine
code, a safe human message, and no raw provider exceptions, PII, or
complaint text ever attached to it.
"""

from __future__ import annotations

# Stable, documented error codes. Extend this set deliberately; do not
# invent ad-hoc codes at call sites.
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"
STT_FAILED = "STT_FAILED"
MODEL_FAILED = "MODEL_FAILED"
TIMEOUT = "TIMEOUT"
INVALID_INPUT = "INVALID_INPUT"


class FusionError(Exception):
    """
    Raised for any Layer 1 failure that must be surfaced safely.

    `message` must never contain raw complaint text, raw PII, or a raw
    provider exception's internals. `details` is for safe, non-sensitive
    debugging context only (e.g. a provider name, a field name) — never
    put user content or exception objects in it.
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_safe_dict(self) -> dict:
        body = {"error": {"code": self.code, "message": self.message}}
        if self.details:
            body["error"]["details"] = self.details
        return body
