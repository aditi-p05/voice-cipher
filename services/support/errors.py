"""
Layer 4B (Support Engine) — structured, safe error handling.

Mirrors services/fusion/errors.py and backend/core/errors.py: a small
typed exception with a stable machine code and a safe human message.
Never attach raw PII, raw transcript text, or raw provider exceptions.
"""

from __future__ import annotations

INVALID_INPUT = "INVALID_INPUT"
SUPPORT_FAILED = "SUPPORT_FAILED"


class SupportError(Exception):
    """Raised for any Support Engine failure that must be surfaced safely."""

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
