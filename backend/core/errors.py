"""
Layer 0 — structured error responses.

No internal stack traces or exception internals are ever returned to the
client. IntakeValidationError -> 400 with a stable machine-readable code.
Anything unexpected -> generic 500 with no internals leaked.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.core.logging import log_event
from backend.ingestion.validator import IntakeValidationError


def error_body(code: str, message: str, field_name: str | None = None) -> dict:
    body = {"error": {"code": code, "message": message}}
    if field_name:
        body["error"]["field"] = field_name
    return body


async def intake_validation_error_handler(
    request: Request, exc: IntakeValidationError
) -> JSONResponse:
    log_event("validation_result", result="failed", error_category=exc.code)
    return JSONResponse(
        status_code=400,
        content=error_body(exc.code, exc.message, exc.field_name),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log_event("validation_result", result="failed", error_category="INTERNAL_ERROR")
    return JSONResponse(
        status_code=500,
        content=error_body("INTERNAL_ERROR", "An unexpected error occurred."),
    )
