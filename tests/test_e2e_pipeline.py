"""
End-to-end integration tests across the full Voice Cipher pipeline.

Scope and honesty note (read before extending this file):

    Layer 0 (Input/Ingestion)      -> backend/            REAL, implemented
    Layer 1 (Fusion/Privacy)       -> services/fusion/     REAL, implemented
    Layer 2 (SVI/Risk)             -> services/svi/        NOT IMPLEMENTED
                                       (reserved, .gitkeep only)
    Layer 4A (Service/RAG)         -> services/rag/        REAL, implemented
    Layer 4B (Support)             -> services/support/    NOT IMPLEMENTED
                                       (reserved, .gitkeep only)
    Layer 5 (Dashboard/Operator)   -> (none yet)            NOT IMPLEMENTED

Per CONTRACTS.md integration rule 5 ("Use mocks/fakes when an
upstream/downstream module is incomplete") and rule 6 ("Add integration
tests at each schema boundary"), the `_Fake*` helpers below are a
TEST-ONLY harness that produces contract-shaped SVIResult /
support-gating / OperatorAction objects so the real Layer 0 -> Layer 1 ->
Layer 4A path can be exercised end-to-end. They are intentionally tiny,
live only in this test file, and are NOT a submission for Member 3's,
Member 5's, or Member 6's module — the integration report calls this out
explicitly as an open gap, not a completed layer.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.ingestion.case_store import InMemoryCaseStore
from backend.ingestion.intake_service import IntakeService
from backend.ingestion.input_envelope import ConsentStatus
from backend.ingestion.validator import IntakeValidationError, RawIntakeRequest
from backend.models.enums import Channel, Modality
from backend.transport.dispatcher import InMemoryDispatcher
from services.fusion import FusionError, build_evidence_bundle
from services.fusion.stt.base import STTResult
from services.fusion.stt.mock_provider import MockSTTProvider
from services.rag.service import (
    RagUnavailableError,
    RagValidationError,
    generate_recommendation,
)

SOP_DIR = "data/sop"

# Markers that, per the integration brief ("CRITICAL safety markers cannot
# be overridden by a lower ML score"), force a CRITICAL tier regardless of
# rule/ML scores.
_CRITICAL_SAFETY_MARKERS = {"self_harm", "weapon", "immediate_danger"}


# ---------------------------------------------------------------------------
# Test-only harness for the not-yet-implemented layers (see module docstring)
# ---------------------------------------------------------------------------
def _fake_calculate_svi(evidence_bundle, rule_floor: int, ml_score: int | None = None) -> dict:
    marker_types = {m.type for m in evidence_bundle.markers}
    has_critical_marker = bool(marker_types & _CRITICAL_SAFETY_MARKERS)

    final_score = max(rule_floor, ml_score) if ml_score is not None else rule_floor
    if has_critical_marker:
        final_score = max(final_score, 80)  # CRITICAL floor cannot be overridden downward

    if final_score >= 80:
        tier = "CRITICAL"
    elif final_score >= 50:
        tier = "HIGH"
    elif final_score >= 25:
        tier = "MODERATE"
    else:
        tier = "LOW"

    return {
        "schema_version": "1.0.0",
        "case_id": evidence_bundle.case_id,
        "svi_score": final_score,
        "risk_tier": tier,
        "rule_floor": rule_floor,
        "ml_score": ml_score,
        "explanation": [{"feature": "harness_rule_floor", "impact": rule_floor}],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class _FakeSviUnavailable(Exception):
    """Test-harness stand-in for a Layer 2 failure (e.g. model outage)."""


def _fake_calculate_svi_failing(evidence_bundle, rule_floor: int, ml_score: int | None = None) -> dict:
    raise _FakeSviUnavailable("SVI model unavailable")


def _fake_support_gate(risk_tier: str) -> dict:
    """LOW/MODERATE may receive support; HIGH is restricted; CRITICAL disables
    automated support as a *primary* intervention (human operator required)."""
    if risk_tier in ("LOW", "MODERATE"):
        return {"support_offered": True, "reason": "eligible_tier"}
    if risk_tier == "HIGH":
        return {"support_offered": False, "reason": "high_risk_restricted"}
    return {"support_offered": False, "reason": "critical_requires_human_operator"}


def _fake_support_gate_failing(risk_tier: str) -> dict:
    raise RuntimeError("support subsystem outage")


def _fake_operator_action(case_id: str, operator_id: str, action: str, **kwargs) -> dict:
    assert action in ("CONFIRM", "OVERRIDE")
    return {
        "schema_version": "1.0.0",
        "case_id": case_id,
        "operator_id": operator_id,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }


def _run_pipeline(envelope, *, stt_provider=None, rule_floor: int, ml_score=None):
    """Layer 1 -> (harness Layer 2) -> Layer 4A, given an already-built envelope."""
    bundle = build_evidence_bundle(envelope, stt_provider=stt_provider)
    svi_result = _fake_calculate_svi(bundle, rule_floor=rule_floor, ml_score=ml_score)
    recommendation = generate_recommendation(bundle, svi_result, sop_directory=SOP_DIR)
    return bundle, svi_result, recommendation


# ---------------------------------------------------------------------------
# 1. Full path: Portal complaint -> ... -> Operator CONFIRM / OVERRIDE
# ---------------------------------------------------------------------------
def test_full_path_portal_complaint_to_operator_confirm_and_override():
    case_store = InMemoryCaseStore()
    dispatcher = InMemoryDispatcher()
    intake_service = IntakeService(case_store=case_store, dispatcher=dispatcher)

    # 1-2. Portal complaint -> InputEnvelope
    raw = RawIntakeRequest(
        channel=Channel.PORTAL.value,
        modalities=[Modality.TEXT.value, Modality.STRUCTURED_DATA.value],
        text_body="He threatened to hurt me if I report this. I am scared.",
        structured_fields={"category": "harassment", "location": "test_area", "fields": {}},
    )
    outcome = intake_service.handle_intake(raw, consent=ConsentStatus(consent_given=True))
    envelope = outcome.envelope
    assert outcome.dispatch.accepted is True

    # 3. EvidenceBundle
    bundle = build_evidence_bundle(envelope)
    assert bundle.pii_redacted is True
    assert bundle.transcript is not None

    # 4-5. SVIResult + risk tier (harness)
    svi_result = _fake_calculate_svi(bundle, rule_floor=55)
    assert svi_result["risk_tier"] == "HIGH"

    # 6. RAG recommendation
    recommendation = generate_recommendation(bundle, svi_result, sop_directory=SOP_DIR)
    assert recommendation["requires_operator_confirmation"] is True
    assert recommendation["recommendations"], "HIGH tier must retrieve an actionable SOP"

    # 7. Support gating
    support = _fake_support_gate(svi_result["risk_tier"])
    assert support["support_offered"] is False  # HIGH is restricted

    # 8. Dashboard case (shape only, since Layer 5 does not exist yet)
    dashboard_case = {
        "case_id": envelope.case_id,
        "risk_tier": svi_result["risk_tier"],
        "recommendation": recommendation,
        "support": support,
    }
    assert dashboard_case["case_id"] == envelope.case_id

    # 9. Operator CONFIRM
    confirm = _fake_operator_action(
        case_id=envelope.case_id,
        operator_id="OP-001",
        action="CONFIRM",
        original_recommendation=recommendation,
        final_decision=recommendation["recommendations"][0]["action"],
    )
    assert confirm["action"] == "CONFIRM"

    # 10. Operator OVERRIDE
    override = _fake_operator_action(
        case_id=envelope.case_id,
        operator_id="OP-001",
        action="OVERRIDE",
        original_recommendation=recommendation,
        final_decision="Escalate to specialised unit instead",
        reason="Operator judgement based on additional context",
    )
    assert override["action"] == "OVERRIDE"
    assert override["final_decision"] != recommendation["recommendations"][0]["action"]

    # AI recommendation is never final on its own: both paths require a
    # recorded human OperatorAction, and the recommendation never bypasses it.
    assert recommendation["requires_operator_confirmation"] is True


# ---------------------------------------------------------------------------
# 2. Risk tiers: LOW / MODERATE / HIGH / CRITICAL, end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rule_floor,ml_score,expected_tier",
    [
        (10, None, "LOW"),
        (30, 20, "MODERATE"),  # max(30, 20) = 30
        (60, None, "HIGH"),
        (85, None, "CRITICAL"),
    ],
)
def test_risk_tiers_map_correctly_end_to_end(rule_floor, ml_score, expected_tier):
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="Routine status update.")
    bundle, svi_result, recommendation = _run_pipeline(envelope, rule_floor=rule_floor, ml_score=ml_score)
    assert svi_result["risk_tier"] == expected_tier
    assert recommendation["case_id"] == envelope.case_id


def test_final_score_uses_max_of_rule_floor_and_ml_score():
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="Neutral text.")
    bundle = build_evidence_bundle(envelope)

    lower_ml = _fake_calculate_svi(bundle, rule_floor=40, ml_score=10)
    assert lower_ml["svi_score"] == 40  # rule_floor wins

    higher_ml = _fake_calculate_svi(bundle, rule_floor=40, ml_score=90)
    assert higher_ml["svi_score"] == 90  # ml_score wins


def test_critical_safety_marker_cannot_be_overridden_by_lower_ml_score():
    envelope = _build_envelope(
        channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="I want to end my life."
    )
    bundle = build_evidence_bundle(envelope)
    assert "self_harm" in {m.type for m in bundle.markers}

    # Even with a very low rule floor and a low ML score, a CRITICAL safety
    # marker must force the CRITICAL tier.
    result = _fake_calculate_svi(bundle, rule_floor=5, ml_score=10)
    assert result["risk_tier"] == "CRITICAL"
    assert result["svi_score"] >= 80


# ---------------------------------------------------------------------------
# 3. Input variety: text-only, voice, portal, multimodal, missing voice feats
# ---------------------------------------------------------------------------
def _build_envelope(*, channel, modalities, text=None, audio=None, structured_data=None, case_id="CASE-E2E0001"):
    from backend.ingestion.input_envelope import (
        AudioReference,
        InputEnvelope,
        StructuredComplaintData,
        TextContent,
    )

    return InputEnvelope(
        case_id=case_id,
        channel=channel,
        modalities=modalities,
        text=TextContent(body=text) if text else None,
        audio=audio,
        structured_data=StructuredComplaintData(**structured_data) if structured_data else None,
    )


def test_text_only_chatbot_end_to_end():
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="I am scared, he threatened me.")
    bundle, svi_result, recommendation = _run_pipeline(envelope, rule_floor=55)
    assert bundle.transcript is not None
    assert recommendation["recommendations"]


def test_voice_channel_with_transcript_end_to_end():
    from backend.ingestion.input_envelope import AudioReference

    envelope = _build_envelope(
        channel=Channel.VOICE_CALL, modalities=[Modality.AUDIO], audio=AudioReference(audio_ref_id="AUD-E2E-1")
    )
    provider = MockSTTProvider(
        fixtures={"AUD-E2E-1": STTResult(text="He hit me and I am terrified.", language="en", confidence=0.9)}
    )
    bundle, svi_result, recommendation = _run_pipeline(envelope, stt_provider=provider, rule_floor=60)
    assert bundle.transcript is not None
    assert recommendation["recommendations"]


def test_voice_channel_missing_optional_transcript_still_completes_pipeline():
    """Real-world default: no STT transcript yet available. Must not break
    Layer 1 -> Layer 4A (this is the regression covered by the pii_redacted
    fix in services/fusion/fusion_service.py)."""
    from backend.ingestion.input_envelope import AudioReference

    envelope = _build_envelope(
        channel=Channel.VOICE_CALL, modalities=[Modality.AUDIO], audio=AudioReference(audio_ref_id="AUD-NO-FIXTURE")
    )
    bundle, svi_result, recommendation = _run_pipeline(envelope, rule_floor=60)
    assert bundle.transcript is None
    assert bundle.pii_redacted is True
    assert recommendation["case_id"] == envelope.case_id


def test_portal_channel_end_to_end():
    envelope = _build_envelope(
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA],
        text="Filing a formal complaint.",
        structured_data={"category": "harassment"},
    )
    bundle, svi_result, recommendation = _run_pipeline(envelope, rule_floor=30)
    assert recommendation["case_id"] == envelope.case_id


def test_multimodal_portal_with_audio_text_and_structured_data_end_to_end():
    from backend.ingestion.input_envelope import AudioReference

    envelope = _build_envelope(
        channel=Channel.PORTAL,
        modalities=[Modality.TEXT, Modality.STRUCTURED_DATA, Modality.AUDIO],
        text="Synthetic portal complaint text.",
        structured_data={"category": "harassment"},
        audio=AudioReference(audio_ref_id="AUD-E2E-MM"),
    )
    provider = MockSTTProvider(
        fixtures={"AUD-E2E-MM": STTResult(text="Synthetic spoken addendum.", language="en", confidence=0.8)}
    )
    bundle, svi_result, recommendation = _run_pipeline(envelope, stt_provider=provider, rule_floor=40)
    assert "Synthetic portal complaint text" in bundle.transcript.text
    assert recommendation["case_id"] == envelope.case_id


# ---------------------------------------------------------------------------
# 4. Failure handling: RAG failure, SVI failure, support failure, invalid input
# ---------------------------------------------------------------------------
def test_rag_failure_is_a_safe_structured_error_not_a_fake_success():
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="Some complaint text.")
    bundle = build_evidence_bundle(envelope)
    svi_result = _fake_calculate_svi(bundle, rule_floor=60)

    class UnavailableStore:
        def search(self, query, top_k=3):
            raise RuntimeError("vector database connection refused")

    with pytest.raises(RagUnavailableError):
        generate_recommendation(bundle, svi_result, store=UnavailableStore())


def test_svi_failure_does_not_silently_produce_a_fake_result():
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="Some complaint text.")
    bundle = build_evidence_bundle(envelope)

    with pytest.raises(_FakeSviUnavailable):
        _fake_calculate_svi_failing(bundle, rule_floor=60)


def test_support_failure_cannot_break_the_critical_service_path():
    """A support-subsystem outage must never prevent the operator-facing
    recommendation from being delivered — support is additive, not on the
    critical path to a recommendation."""
    envelope = _build_envelope(channel=Channel.CHATBOT, modalities=[Modality.TEXT], text="Needs review.")
    bundle, svi_result, recommendation = _run_pipeline(envelope, rule_floor=30)

    try:
        _fake_support_gate_failing(svi_result["risk_tier"])
        support_status = {"support_offered": False, "reason": "unexpected_success"}
    except RuntimeError:
        # Caller degrades to "no automated support offered" rather than
        # propagating the failure into the recommendation path.
        support_status = {"support_offered": False, "reason": "support_unavailable"}

    assert recommendation["recommendations"], "recommendation must still be delivered"
    assert support_status["support_offered"] is False


def test_invalid_input_is_rejected_before_evidence_bundle_is_built():
    case_store = InMemoryCaseStore()
    dispatcher = InMemoryDispatcher()
    intake_service = IntakeService(case_store=case_store, dispatcher=dispatcher)

    raw = RawIntakeRequest(channel="not_a_real_channel", modalities=["text"], text_body="hello")
    with pytest.raises(IntakeValidationError) as exc_info:
        intake_service.handle_intake(raw)
    assert exc_info.value.code == "UNSUPPORTED_CHANNEL"


def test_invalid_evidence_bundle_input_is_rejected_by_fusion():
    with pytest.raises(FusionError) as exc_info:
        build_evidence_bundle("not an envelope")  # type: ignore[arg-type]
    assert exc_info.value.code == "INVALID_INPUT"


def test_malformed_evidence_is_rejected_by_rag_before_retrieval():
    with pytest.raises(RagValidationError):
        generate_recommendation({"case_id": "CASE-X"}, {"case_id": "CASE-X", "risk_tier": "LOW"})
