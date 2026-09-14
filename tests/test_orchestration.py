from __future__ import annotations

import time

import pytest

from backend.ingestion.input_envelope import (
    AudioReference,
    InputEnvelope,
    StructuredComplaintData,
    TextContent,
)
from backend.models.enums import Channel, Modality
from backend.orchestration.adapters.svi_adapter import MockSVIProvider
from backend.orchestration.errors import EXTERNAL_SERVICE_FAILED
from backend.orchestration.nodes import Providers
from backend.orchestration.run import run_pipeline
from services.fusion.errors import FusionError
from services.support.errors import SupportError


# ---------------------------------------------------------------------------
# Envelope builders for each input path
# ---------------------------------------------------------------------------

def chatbot_envelope(text: str, case_id: str = "CASE-CHAT01") -> InputEnvelope:
    return InputEnvelope(
        case_id=case_id,
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        text=TextContent(body=text),
    )


def portal_envelope(text: str, case_id: str = "CASE-PORTAL1") -> InputEnvelope:
    return InputEnvelope(
        case_id=case_id,
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA],
        text=TextContent(body=text),
        structured_data=StructuredComplaintData(category="harassment", fields={}),
    )


def voice_envelope(case_id: str = "CASE-VOICE01") -> InputEnvelope:
    return InputEnvelope(
        case_id=case_id,
        channel=Channel.VOICE_CALL,
        modalities=[Modality.AUDIO, Modality.TEXT],
        audio=AudioReference(audio_ref_id="AR-1", mime_type="audio/wav"),
    )


def multimodal_portal_envelope(text: str, case_id: str = "CASE-MULTI01") -> InputEnvelope:
    return InputEnvelope(
        case_id=case_id,
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO],
        text=TextContent(body=text),
        structured_data=StructuredComplaintData(category="threat", fields={}),
        audio=AudioReference(audio_ref_id="AR-2", mime_type="audio/wav"),
    )


# ---------------------------------------------------------------------------
# Full successful pipeline / per-input-path flows
# ---------------------------------------------------------------------------

def test_full_successful_pipeline_text_only():
    case = run_pipeline(chatbot_envelope("I am scared they will come back to my house."))
    assert case["evidence_available"] is True
    assert case["svi_result"] is not None
    assert case["risk_tier"] in {"LOW", "MODERATE", "HIGH", "CRITICAL"}
    assert case["service_recommendation"] is not None
    assert case["service_recommendation"]["requires_operator_confirmation"] is True
    assert case["support_decision"] is not None
    assert case["errors"] == []


def test_portal_flow_with_structured_data():
    case = run_pipeline(portal_envelope("The complaint concerns repeated intimidation."))
    assert case["evidence_available"] is True
    assert case["support_decision"] is not None
    assert case["errors"] == []


def test_voice_flow_with_no_registered_transcript_still_completes_safely():
    # MockSTTProvider never fabricates a transcript for an unregistered
    # audio_ref_id -- evidence should still be produced (no crash), just
    # with no transcript/markers.
    #
    # NOTE: an earlier version of this test asserted RAG failed closed
    # here, because fusion previously set `pii_redacted=was_redacted`,
    # which was False whenever there was no transcript at all -- treating
    # "nothing existed to redact" the same as "redaction never happened".
    # That was fixed upstream (feature/integration-fixes,
    # services/fusion/fusion_service.py): `pii_redacted` is now correctly
    # True whenever there is no un-redacted raw text outstanding,
    # including the no-transcript case. RAG now runs even on sparse
    # evidence, closing exactly the gap flagged in the earlier
    # integration report.
    case = run_pipeline(voice_envelope())
    assert case["evidence_available"] is True
    assert case["svi_result"]["risk_tier"] == "LOW"
    assert case["errors"] == []
    assert case["service_recommendation"] is not None
    assert case["support_decision"] is not None


def test_multimodal_portal_flow():
    case = run_pipeline(multimodal_portal_envelope("He said he would hurt me if I told anyone."))
    assert case["evidence_available"] is True
    assert case["service_recommendation"] is not None
    assert case["support_decision"] is not None


# ---------------------------------------------------------------------------
# Risk-tier-driven support gating, exercised through the real graph with a
# fake SVI provider (SVI/Risk itself is Member 3's unimplemented module).
# ---------------------------------------------------------------------------

class _FixedSVIProvider:
    def __init__(self, tier: str, score: int):
        self._tier = tier
        self._score = score

    def calculate(self, evidence_bundle):
        # Real EvidenceBundle object -- attribute access, not dict keys.
        return {
            "schema_version": "1.0.0",
            "case_id": evidence_bundle.case_id,
            "svi_score": self._score,
            "risk_tier": self._tier,
            "rule_floor": 80 if self._tier == "CRITICAL" else 0,
            "ml_score": self._score,
            "explanation": [],
            "created_at": "2026-01-01T00:00:00+00:00",
        }


@pytest.mark.parametrize("tier,score", [("LOW", 10), ("MODERATE", 40)])
def test_low_and_moderate_risk_support_enabled_end_to_end(tier, score):
    providers = Providers(svi_provider=_FixedSVIProvider(tier, score))
    case = run_pipeline(chatbot_envelope("A short, low-signal message."), providers=providers)
    assert case["risk_tier"] == tier
    assert case["support_decision"]["support_enabled"] is True
    assert case["service_recommendation"] is not None  # service still runs at every tier


def test_high_risk_support_restricted_end_to_end():
    providers = Providers(svi_provider=_FixedSVIProvider("HIGH", 65))
    case = run_pipeline(chatbot_envelope("A concerning message."), providers=providers)
    assert case["risk_tier"] == "HIGH"
    assert case["support_decision"]["support_enabled"] is False
    assert case["support_decision"]["mode"] == "restricted"
    assert case["service_recommendation"] is not None


def test_critical_risk_support_disabled_end_to_end():
    providers = Providers(svi_provider=_FixedSVIProvider("CRITICAL", 92))
    case = run_pipeline(chatbot_envelope("An urgent, high-risk message."), providers=providers)
    assert case["risk_tier"] == "CRITICAL"
    assert case["support_decision"]["support_enabled"] is False
    assert case["support_decision"]["mode"] == "disabled"
    # Critical path (service/escalation) must proceed regardless of support.
    assert case["service_recommendation"] is not None
    assert case["service_recommendation"]["requires_operator_confirmation"] is True


def test_critical_marker_cannot_be_overridden_by_low_ml_score():
    # Real fusion text hitting a high-confidence self_harm cue, scored by
    # the now-real services.svi.svi_service.calculate_svi (Member 3).
    # Its rule engine floors self_harm at 90.0 -- verify that floor wins
    # over whatever the (low) ML/lexical score would otherwise be, and
    # that CRITICAL gating still disables automated support.
    case = run_pipeline(chatbot_envelope("I want to end my life."))
    assert case["svi_result"]["rule_floor"] >= 80  # CRITICAL-band floor per CONTRACTS.md 6.3
    assert case["svi_result"]["svi_score"] >= 80
    assert case["risk_tier"] == "CRITICAL"
    assert case["support_decision"]["support_enabled"] is False


# ---------------------------------------------------------------------------
# Required failure modes
# ---------------------------------------------------------------------------

def test_fusion_failure_produces_structured_error_and_skips_downstream():
    def failing_fusion(_envelope):
        raise FusionError("MODEL_FAILED", "Evidence fusion failed unexpectedly.")

    providers = Providers(fusion_fn=failing_fusion)
    case = run_pipeline(chatbot_envelope("text"), providers=providers)

    assert case["evidence_available"] is False
    assert case["svi_result"] is None
    assert case["service_recommendation"] is None
    assert case["support_decision"] is None
    assert any(e["code"] == "FUSION_FAILED" for e in case["errors"])


def test_svi_failure_produces_structured_error_and_skips_downstream():
    class BrokenSVI:
        def calculate(self, evidence_bundle):
            raise RuntimeError("boom")

    providers = Providers(svi_provider=BrokenSVI())
    case = run_pipeline(chatbot_envelope("text"), providers=providers)

    assert case["evidence_available"] is True
    assert case["svi_result"] is None
    assert case["service_recommendation"] is None
    assert case["support_decision"] is None
    assert any(e["code"] == "SVI_FAILED" for e in case["errors"])


def test_svi_never_fabricates_a_score_on_malformed_evidence():
    # Directly exercise the adapter with a malformed EvidenceBundle mapping.
    from backend.orchestration.adapters.svi_adapter import run_svi

    result, error = run_svi({"case_id": "CASE-X"})  # missing required fields
    assert result is None
    assert error["code"] == "SVI_FAILED"


def test_rag_failure_does_not_block_support_or_pipeline_completion():
    def failing_rag(_evidence, _svi):
        raise RuntimeError("SOP store unavailable")

    providers = Providers(rag_fn=failing_rag)
    case = run_pipeline(chatbot_envelope("I am scared."), providers=providers)

    assert case["service_recommendation"] is None
    assert any(e["code"] == "RAG_FAILED" for e in case["errors"])
    # Support still ran even though RAG failed.
    assert case["support_decision"] is not None


def test_support_failure_does_not_block_service_or_pipeline_completion():
    def failing_support(_evidence, _svi):
        raise SupportError("SUPPORT_FAILED", "Support decision failed unexpectedly.")

    providers = Providers(support_fn=failing_support)
    case = run_pipeline(chatbot_envelope("I am scared."), providers=providers)

    assert case["support_decision"] is None
    assert any(e["code"] == "SUPPORT_FAILED" for e in case["errors"])
    # Service/critical path must proceed regardless of support failing.
    assert case["service_recommendation"] is not None


def test_critical_path_continues_despite_optional_support_failure():
    def failing_support(_evidence, _svi):
        raise RuntimeError("support subsystem down")

    providers = Providers(
        svi_provider=_FixedSVIProvider("CRITICAL", 95),
        support_fn=failing_support,
    )
    case = run_pipeline(chatbot_envelope("An urgent message."), providers=providers)

    assert case["risk_tier"] == "CRITICAL"
    assert case["support_decision"] is None
    assert any(e["code"] == "SUPPORT_FAILED" for e in case["errors"])
    # The critical service/escalation recommendation must still be present.
    assert case["service_recommendation"] is not None
    assert case["service_recommendation"]["requires_operator_confirmation"] is True


def test_timeout_produces_structured_timeout_error():
    def slow_fusion(_envelope):
        time.sleep(0.2)
        raise AssertionError("should have timed out before returning")

    providers = Providers(fusion_fn=slow_fusion, fusion_timeout_seconds=0.01)
    case = run_pipeline(chatbot_envelope("text"), providers=providers)

    assert case["evidence_available"] is False
    assert any(e["code"] == "TIMEOUT" for e in case["errors"])


def test_malformed_shared_object_from_rag_layer_is_handled_safely():
    def malformed_rag(_evidence, _svi):
        return {"unexpected": "shape"}  # not schema-shaped, but not an exception either

    providers = Providers(rag_fn=malformed_rag)
    case = run_pipeline(chatbot_envelope("text"), providers=providers)

    # Orchestration must not crash on a malformed-but-non-raising result;
    # it is passed through as-is and `requires_operator_confirmation`
    # correctly reflects that this is not a valid recommendation.
    assert case["service_recommendation"] == {"unexpected": "shape"}
    assert case["requires_operator_confirmation"] is False


def test_rag_and_support_failures_in_the_same_run_are_both_preserved():
    # This is the scenario the errors reducer exists for: SERVICE and
    # SUPPORT are true parallel branches off SVI. Before the
    # `operator.add` reducer, two nodes writing "errors" in the same
    # superstep would have the default overwrite behavior silently drop
    # one branch's error. Both must survive here.
    def failing_rag(_evidence, _svi):
        raise RuntimeError("SOP store unavailable")

    def failing_support(_evidence, _svi):
        raise RuntimeError("support subsystem down")

    providers = Providers(rag_fn=failing_rag, support_fn=failing_support)
    case = run_pipeline(chatbot_envelope("I am scared."), providers=providers)

    codes = {e["code"] for e in case["errors"]}
    assert "RAG_FAILED" in codes
    assert "SUPPORT_FAILED" in codes
    assert case["service_recommendation"] is None
    assert case["support_decision"] is None



    from backend.orchestration.adapters.fusion_adapter import run_fusion

    bundle, error = run_fusion(object())  # not an InputEnvelope
    assert bundle is None
    assert error["code"] == "FUSION_FAILED"
