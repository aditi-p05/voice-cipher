"""Public Layer 4A adapter: EvidenceBundle + SVIResult -> ServiceRecommendation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .documents import chunk_sop_document, load_sop_documents
from .generators import DeterministicRecommendationGenerator, RecommendationGenerator
from .store import InMemoryVectorStore, VectorStore


class RagValidationError(ValueError):
    """Safe validation failure at the Evidence/SVI boundary."""


class RagUnavailableError(RuntimeError):
    """Safe dependency failure; callers must not expose provider internals."""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    raise RagValidationError(f"{label} must be a mapping or Pydantic model")


def _query(evidence: Mapping[str, Any], svi: Mapping[str, Any]) -> str:
    transcript = evidence.get("transcript") or {}
    text = transcript.get("text", "") if isinstance(transcript, Mapping) else ""
    markers = evidence.get("markers", [])
    marker_text = " ".join(f"{m.get('type', '')} {m.get('value', '')}" for m in markers if isinstance(m, Mapping))
    return f"risk tier {svi['risk_tier']} {marker_text} {text}".strip()


def build_store(sop_directory: str | Path) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    for path, content in load_sop_documents(sop_directory):
        store.add(chunk_sop_document(path, content))
    return store


def generate_recommendation(
    evidence_bundle: Any,
    svi_result: Any,
    *,
    store: VectorStore | None = None,
    sop_directory: str | Path = "data/sop",
    generator: RecommendationGenerator | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Create a contract-shaped, operator-gated recommendation from local SOP matches."""
    evidence, svi = _as_mapping(evidence_bundle, "EvidenceBundle"), _as_mapping(svi_result, "SVIResult")
    for field in ("schema_version", "case_id", "source_input_ids", "markers", "pii_redacted", "created_at"):
        if field not in evidence:
            raise RagValidationError(f"EvidenceBundle missing required field: {field}")
    for field in ("schema_version", "case_id", "svi_score", "risk_tier", "rule_floor", "explanation", "created_at"):
        if field not in svi:
            raise RagValidationError(f"SVIResult missing required field: {field}")
    if evidence["case_id"] != svi["case_id"]:
        raise RagValidationError("EvidenceBundle and SVIResult case_id values must match")
    if not evidence["pii_redacted"]:
        raise RagValidationError("EvidenceBundle must be PII-redacted before RAG retrieval")
    if svi["risk_tier"] not in {"LOW", "MODERATE", "HIGH", "CRITICAL"}:
        raise RagValidationError("SVIResult risk_tier is invalid")
    active_store = store if store is not None else build_store(sop_directory)
    query = _query(evidence, svi)
    try:
        matches = active_store.search(query, top_k=top_k)
    except Exception as error:
        raise RagUnavailableError("SOP vector store is unavailable") from error
    selected_generator = generator or DeterministicRecommendationGenerator()
    try:
        recommendations = selected_generator.generate(matches, query)
    except Exception:
        # A provider outage never removes the safe local demo path.
        recommendations = DeterministicRecommendationGenerator().generate(matches, query)
    return {"schema_version": "1.0.0", "case_id": evidence["case_id"], "recommendations": recommendations,
            "requires_operator_confirmation": True, "created_at": datetime.now(timezone.utc).isoformat()}
