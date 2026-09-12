from pathlib import Path

import pytest

from services.rag.documents import chunk_sop_document, load_sop_documents
from services.rag.service import RagValidationError, build_store, generate_recommendation
from services.rag.store import InMemoryVectorStore


SOP = Path(__file__).parents[1] / "data" / "sop"


def evidence(**overrides):
    result = {"schema_version": "1.0.0", "case_id": "CASE-TEST", "source_input_ids": ["IN-1"],
              "transcript": {"text": "synthetic threat marker"}, "markers": [{"type": "threat", "value": "risk", "confidence": .9}],
              "sentiment": None, "voice_features": None, "pii_redacted": True, "created_at": "2026-01-01T00:00:00+00:00"}
    result.update(overrides); return result


def svi(tier="HIGH", **overrides):
    result = {"schema_version": "1.0.0", "case_id": "CASE-TEST", "svi_score": 60, "risk_tier": tier,
              "rule_floor": 50, "ml_score": None, "explanation": [], "created_at": "2026-01-01T00:00:00+00:00"}
    result.update(overrides); return result


def test_loading_chunking_and_metadata():
    docs = load_sop_documents(SOP); assert docs
    chunks = chunk_sop_document(*docs[0]); assert chunks and chunks[0].metadata["version"] == "1.0"
    assert "DEMO-OPERATOR-SUPPORT" in chunks[0].citation


@pytest.mark.parametrize("tier", ["CRITICAL", "HIGH", "MODERATE", "LOW"])
def test_retrieval_and_generation_for_each_risk(tier):
    result = generate_recommendation(evidence(), svi(tier), sop_directory=SOP)
    assert result["requires_operator_confirmation"] is True
    assert result["recommendations"]
    assert "v1.0" in result["recommendations"][0]["source"]
    assert tier.title() in result["recommendations"][0]["source"]


def test_no_matching_sop_returns_empty_recommendations():
    assert generate_recommendation(evidence(transcript={"text": "xyzzy"}, markers=[]), svi(), store=InMemoryVectorStore())["recommendations"] == []


def test_missing_svi_and_malformed_evidence_are_rejected():
    with pytest.raises(RagValidationError): generate_recommendation(evidence(), {"case_id": "CASE-TEST"})
    with pytest.raises(RagValidationError): generate_recommendation({"case_id": "CASE-TEST"}, svi())


def test_vector_store_and_llm_fallback_behaviour():
    store = build_store(SOP); assert store.search("high risk priority", 1)
    class UnavailableLlm:
        def generate(self, context, query): raise RuntimeError("provider unavailable")
    assert generate_recommendation(evidence(), svi(), store=store, generator=UnavailableLlm())["recommendations"]


def test_vector_store_unavailable_is_a_safe_error():
    class UnavailableStore:
        def search(self, query, top_k=3): raise RuntimeError("database unavailable")
    from services.rag.service import RagUnavailableError
    with pytest.raises(RagUnavailableError, match="vector store is unavailable"):
        generate_recommendation(evidence(), svi(), store=UnavailableStore())


def test_unredacted_evidence_is_rejected():
    with pytest.raises(RagValidationError): generate_recommendation(evidence(pii_redacted=False), svi())
