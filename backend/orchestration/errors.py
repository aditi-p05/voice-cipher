"""
Layer 4B (Orchestration) — structured, safe error handling.

Same pattern as services/fusion/errors.py and services/support/errors.py.
These codes are what appear in PipelineState.errors and, eventually, in
whatever the dashboard shows an operator — never a raw exception.
"""

from __future__ import annotations

FUSION_FAILED = "FUSION_FAILED"
SVI_FAILED = "SVI_FAILED"
RAG_FAILED = "RAG_FAILED"
SUPPORT_FAILED = "SUPPORT_FAILED"
TIMEOUT = "TIMEOUT"
EXTERNAL_SERVICE_FAILED = "EXTERNAL_SERVICE_FAILED"
INVALID_STATE = "INVALID_STATE"

STAGE_CODES = {
    "fusion": FUSION_FAILED,
    "svi": SVI_FAILED,
    "rag": RAG_FAILED,
    "support": SUPPORT_FAILED,
}


class OrchestrationError(Exception):
    """Raised only for genuinely unrecoverable orchestration-level failures."""

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


def stage_error(stage: str, message: str, *, code: str | None = None) -> dict:
    """Build a structured, safe error record for PipelineState.errors."""
    return {"stage": stage, "code": code or STAGE_CODES.get(stage, EXTERNAL_SERVICE_FAILED), "message": message}
