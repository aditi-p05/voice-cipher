"""
Layer 0 — application entrypoint.

Run with: uvicorn backend.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router as layer0_router
from backend.core.errors import intake_validation_error_handler, unhandled_error_handler
from backend.ingestion.validator import IntakeValidationError

app = FastAPI(
    title="NHAA — Layer 0 Input Layer",
    description="Ingestion foundation for voice call, chatbot, and complaint portal intake.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(layer0_router)

app.add_exception_handler(IntakeValidationError, intake_validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "layer": "layer0-input"}
