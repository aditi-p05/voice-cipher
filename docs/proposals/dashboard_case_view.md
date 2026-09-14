# Proposal: `DashboardCaseView` as a CONTRACTS.md schema

**Status:** proposed, not yet adopted. Per CONTRACTS.md section 15
("Propose -> Discuss -> Update CONTRACTS.md -> Implement -> Test ->
Merge"), this is the "Propose" step. Do not treat this document as a
contract until it is discussed and folded into CONTRACTS.md itself.

## Why this is needed

`backend/orchestration/run.py::build_dashboard_case_view` already
aggregates `EvidenceBundle`, `SVIResult`, `ServiceRecommendation`, and
Layer 4B's own `SupportDecision` into one dict for Member 6 to consume.
That shape currently exists only as this module's return value -- it is
not a numbered CONTRACTS.md schema, so Member 6 has nothing stable to
build the dashboard/API integration against, and any orchestration-side
change to it would silently break Member 6's code with no contract to
catch it (violating CONTRACTS.md section 12, rule 2).

## Proposed schema (`DashboardCaseView`)

Producer: orchestration (`backend/orchestration/run.py`).
Consumer: Member 6 (dashboard/API).

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | `"1.0.0"` |
| `case_id` | string | From the envelope, or the merged EvidenceBundle when there is no single canonical envelope (e.g. a voice session) |
| `input_id` | string or null | Latest contributing input; null for a merged voice session with no single input |
| `channel` | string | `voice_call`, `chatbot`, `portal`, or `voice_call_session` for a merged multi-chunk voice case |
| `evidence_available` | boolean | False only if Fusion failed entirely |
| `evidence_bundle` | object or null | The real `EvidenceBundle`, as-is |
| `svi_result` | object or null | The real `SVIResult`, as-is |
| `risk_tier` | string or null | Convenience mirror of `svi_result.risk_tier` |
| `service_recommendation` | object or null | The real `ServiceRecommendation`, as-is |
| `support_decision` | object or null | Layer 4B's `SupportDecision` (not yet a CONTRACTS.md schema itself -- see open question below) |
| `requires_operator_confirmation` | boolean | Derived strictly from `service_recommendation.requires_operator_confirmation`; never true for a malformed/absent recommendation |
| `errors` | list of objects | Structured `{stage, code, message}` records accumulated from any stage; empty list on a fully clean run |

## Open questions for the team

1. Should `SupportDecision` (services/support/models.py) also become a
   numbered CONTRACTS.md schema (6.6), given the dashboard now depends
   on its shape too?
2. Is `channel: "voice_call_session"` an acceptable synthetic value, or
   should merged voice sessions carry the original `"voice_call"` value
   plus a separate `is_session: true` flag?
3. Does Member 6 need per-chunk case views during an in-progress voice
   call, or only the final one? (Today, `voice_session.py` returns a
   fresh view on every chunk -- nothing stops Member 6 from polling it,
   but nothing formalizes that as a contract either.)

## Non-goals of this proposal

This does not propose a persistence/audit store, nor does it change any
existing CONTRACTS.md schema (`InputEnvelope`, `EvidenceBundle`,
`SVIResult`, `ServiceRecommendation`, `OperatorAction`). It only proposes
formalizing the aggregate shape that already exists in code today.
