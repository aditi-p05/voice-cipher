# Team Voice Cipher — Shared Integration Contracts

**Status:** initial contract, based on the existing Layer 0 implementation. This document is the single source of truth for interfaces shared by the six team members. It does not prescribe internal file layouts.

## 1. Project overview

Team Voice Cipher is building an AI-assisted multimodal decision-support system for SIH 2026 Problem Statement 26093, NHAA (14566). The system ingests voice, chat, and complaint-portal interactions, prepares evidence, determines a vulnerability/stress index (SVI) and risk tier, retrieves procedural guidance, and presents recommendations for **human operator confirmation**.

The pipeline is:

```text
Input -> InputEnvelope -> EvidenceBundle -> SVIResult -> Risk tier
      -> ServiceRecommendation / Support -> Dashboard -> OperatorAction
```

Automation must not make autonomous medical, legal, policing, or therapeutic decisions. Operator action remains authoritative.

## 2. Architecture and module boundary

| Stage | Shared object | Producer | Consumer |
| --- | --- | --- | --- |
| Input / Ingestion | `InputEnvelope` | Member 1 | Member 2; orchestration adapter |
| Fusion / Privacy | `EvidenceBundle` | Member 2 | Member 3; Member 4 |
| SVI / Risk | `SVIResult` | Member 3 | Member 4; Member 5; Member 6 |
| Service / Support | `ServiceRecommendation` | Member 4 or Member 5 | Member 6 / operator |
| Dashboard decision | `OperatorAction` | Member 6 / human operator | audit and orchestration |

Current implementation is only the first row. Later modules consume a shared object rather than importing another member's internal helpers.

## 3. Ownership

| Member | Module | Responsibility | Initial ownership location |
| --- | --- | --- | --- |
| Member 1 | Input / Ingestion | Voice, chat, portal, `InputEnvelope` | `backend/ingestion/`, `backend/api/`, `backend/routing/`, `backend/transport/` |
| Member 2 | Fusion / Privacy | STT, language, PII redaction, evidence extraction | `services/fusion/` |
| Member 3 | SVI / Risk | Rules, ML, risk tier, explainability | `services/svi/` |
| Member 4 | Service / RAG | SOP retrieval and recommendations | `services/rag/` |
| Member 5 | Orchestration / Support | LangGraph, gated support, queues, error handling | `services/support/`, `backend/orchestration/` when introduced |
| Member 6 | API / Dashboard / Integration | FastAPI integration, React, cases, audit | `frontend/react-app/`, API integration code by agreement |

Internal implementation is independent. Shared contracts are not.

## 4. Repository structure

```text
voice-cipher/
├── backend/                 # Current working Layer 0 backend — Member 1
│   ├── api/                 # FastAPI wire schemas, routes, dependencies
│   ├── core/                # Layer 0 errors and safe structured logging
│   ├── ingestion/           # Envelope, validation, case/session abstractions
│   ├── models/              # Current channel/modality enums
│   ├── routing/             # Deterministic modality router
│   ├── transport/           # Dispatcher interface and in-memory adapter
│   └── main.py              # FastAPI entry point
├── services/
│   ├── fusion/              # Reserved: Member 2
│   ├── svi/                 # Reserved: Member 3
│   ├── rag/                 # Reserved: Member 4
│   └── support/             # Reserved: Member 5
├── frontend/react-app/      # Reserved: Member 6
├── data/sample_cases/       # Synthetic, non-sensitive fixtures only
├── data/sop/                # Approved, non-sensitive SOP material only
├── docs/                    # Architecture notes
├── tests/                   # Contract and integration tests
├── CONTRACTS.md             # This contract
└── README.md
```

`backend/orchestration/` is intentionally not created until Member 5 needs a real implementation. Empty reserved folders contain only `.gitkeep`; they are not fake services.

## 5. Naming conventions

- Python files, folders, functions, variables, JSON fields, and API path segments: `snake_case`.
- Classes and Pydantic models: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Enums serialize to their lower-case string values, for example `"voice_call"`.
- New endpoints use plural resources where practical and kebab-free paths, for example `/case/{case_id}/confirm`.
- Shared fields, endpoints, filenames, models, and public functions must not be casually renamed. Add compatibility only after a documented contract change.

The currently accepted public Layer 0 names include `InputEnvelope`, `InputDispatcher.publish`, `route`, `Channel`, `Modality`, `case_id`, `input_id`, `language_hint`, `structured_data`, and `schema_version`.

## 6. Shared schemas

### 6.1 `InputEnvelope` — **implemented**

**Purpose:** normalized, validated Layer 0 evidence intake. **Producer:** Member 1. **Consumers:** Member 2 and future orchestration. This is a Python Pydantic model at `backend.ingestion.input_envelope.InputEnvelope`.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | yes, default `"1.0.0"` | Version of this envelope shape |
| `input_id` | string | yes, generated | Current form: `IN-` plus an uppercase random suffix |
| `case_id` | string | yes | Current validator accepts non-empty IDs beginning `CASE-` |
| `channel` | `Channel` | yes | `voice_call`, `chatbot`, or `portal` |
| `modalities` | list of `Modality` | yes | Non-empty allowed evidence types |
| `timestamp` | timezone-aware datetime | yes, generated | UTC by default |
| `language_hint` | string or null | no | Hint only; Layer 0 does not identify language |
| `text` | `TextContent` or null | conditional | `{ "body": string }` for `text` modality |
| `audio` | `AudioReference` or null | conditional | Reference/metadata only; never raw audio bytes |
| `structured_data` | `StructuredComplaintData` or null | conditional | Portal fields |
| `consent` | `ConsentStatus` | yes, defaults empty | Declared status; not verified by Layer 0 |
| `metadata` | `EnvelopeMetadata` | yes, defaults empty | Operational non-PII metadata only |

`AudioReference` uses `audio_ref_id`, optional `call_id`, `mime_type`, `duration_seconds`, `sample_rate_hz`, `is_chunk`, `sequence_number`, and `storage_uri`. `StructuredComplaintData` uses optional `category`, optional `location`, and a string-to-string `fields` map. `ConsentStatus` uses optional `consent_given` and `consent_source`.

Example portal envelope after intake:

```json
{
  "schema_version": "1.0.0",
  "input_id": "IN-1A2B3C4D5E",
  "case_id": "CASE-01AB23CD",
  "channel": "portal",
  "modalities": ["text", "structured_data"],
  "timestamp": "2026-09-11T10:15:30+00:00",
  "language_hint": "hi",
  "text": {"body": "Synthetic test complaint"},
  "audio": null,
  "structured_data": {"category": "test", "location": "test_area", "fields": {}},
  "consent": {"consent_given": true, "consent_source": "portal_form"},
  "metadata": {"source_ip_hash": null, "user_agent": null, "channel_session_id": null, "extra": {}}
}
```

### 6.2 `EvidenceBundle` — **planned contract**

**Purpose:** Layer 1's redacted, multimodal evidence output. **Producer:** Member 2. **Consumers:** Members 3 and 4. Do not implement this model until Member 2 starts the task.

Required fields: `schema_version` (string), `case_id` (string), `source_input_ids` (list of string), `transcript` (object or null), `markers` (list), `sentiment` (object or null), `voice_features` (object or null), `pii_redacted` (boolean), `created_at` (UTC timestamp).

Optional `transcript` fields: `text`, `language`, `confidence`, `segments`. Each marker uses `type`, `value`, and `confidence`. The bundle must contain redacted text only; downstream teams must not expect raw source text or raw audio.

```json
{
  "schema_version": "1.0.0",
  "case_id": "CASE-01AB23CD",
  "source_input_ids": ["IN-1A2B3C4D5E"],
  "transcript": {"text": "[REDACTED]", "language": "hi", "confidence": 0.91, "segments": []},
  "markers": [{"type": "threat", "value": "retaliation", "confidence": 0.88}],
  "sentiment": {"valence": -0.72, "arousal": 0.83},
  "voice_features": null,
  "pii_redacted": true,
  "created_at": "2026-09-11T10:16:00+00:00"
}
```

### 6.3 `SVIResult` — **planned contract**

**Purpose:** explainable risk output. **Producer:** Member 3. **Consumers:** Members 4–6. Required fields: `schema_version`, `case_id`, `svi_score` (number 0–100), `risk_tier`, `rule_floor` (number 0–100), `ml_score` (number 0–100 or null), `explanation` (list), `created_at`.

Allowed `risk_tier` values: `LOW` (0–24), `MODERATE` (25–49), `HIGH` (50–79), `CRITICAL` (80–100). The intended calculation is `max(rule_floor, ml_score)` when an ML score exists; Member 3 owns the final validated rule.

```json
{
  "schema_version": "1.0.0",
  "case_id": "CASE-01AB23CD",
  "svi_score": 87,
  "risk_tier": "CRITICAL",
  "rule_floor": 80,
  "ml_score": 87,
  "explanation": [{"feature": "explicit_threat", "impact": 31}],
  "created_at": "2026-09-11T10:17:00+00:00"
}
```

### 6.4 `ServiceRecommendation` — **planned contract**

**Purpose:** traceable procedural recommendation. **Producer:** Member 4, with orchestration coordination by Member 5. **Consumer:** Member 6/operator. Required fields: `schema_version`, `case_id`, `recommendations` (list), `requires_operator_confirmation` (boolean), `created_at`.

Each recommendation requires `action`, `reason`, `source`, and `confidence`; `source` must identify the SOP citation/version. Recommendations are never an autonomous final action.

### 6.5 `OperatorAction` — **planned contract**

**Purpose:** audit a human's confirm/override decision. **Producer:** Member 6/human operator. **Consumers:** audit and orchestration. Required fields: `schema_version`, `case_id`, `operator_id`, `action` (`CONFIRM` or `OVERRIDE`), `timestamp`. Optional: `original_recommendation`, `final_decision`, `reason`.

## 7. Channel versus modality

`channel` is **where** the interaction originated. `modalities` are **what evidence** is present. They are separate fields and must remain separate.

| Channel | Allowed modalities in current code | Typical example |
| --- | --- | --- |
| `voice_call` | `audio`, `text` | Audio chunk with optional provider-supplied interim text |
| `chatbot` | `text` | Chat message |
| `portal` | `text`, `structured_data`, `audio` | Form text/fields, optionally audio reference |

The current validator accepts a non-empty subset of a channel's allowed modalities. A portal with audio is still `channel: "portal"`, not a new channel.

## 8. Module interfaces

Current Layer 0 API routes construct `RawIntakeRequest`, then use:

```text
validate_intake(raw) -> ValidatedIntake
IntakeService.handle_intake(...) -> IntakeOutcome
route(input_envelope) -> RoutingResult
InputDispatcher.publish(input_envelope) -> DispatchResult
```

`RoutingResult` exposes `available_modalities`, `has_audio`, `has_text`, `has_structured_data`, and `transcript_expected`. It contains no risk or escalation fields.

The planned cross-module flow is:

```text
InputEnvelope
  -> build_evidence_bundle(input_envelope) [Member 2: planned]
  -> EvidenceBundle
  -> calculate_svi(evidence_bundle) [Member 3: planned]
  -> SVIResult
  -> generate_recommendation(evidence_bundle, svi_result) [Member 4: planned]
  -> ServiceRecommendation
```

Function names after Layer 0 are proposed interface names, not existing functions. Teams must document the actual public adapter names here before integration.

## 9. API contracts

### Currently implemented (Layer 0)

| Endpoint | Request model | Response | Notes |
| --- | --- | --- | --- |
| `POST /chat/message` | `ChatMessageIn` | `IntakeAck` | `message` is required; optional `case_id`, `language_hint`, `channel_session_id` |
| `POST /portal/submit` | `PortalSubmitIn` | `IntakeAck` | accepts optional `complaint_text`, `structured_data`, `audio`, `language_hint`, `consent_given`; request must yield at least one valid modality |
| `POST /voice/incoming` | `VoiceIncomingIn` | `VoiceSessionAck` | required `call_id`; establishes call-to-case session mapping only; it does not create an envelope or dispatch evidence |
| `POST /voice/chunk` | `VoiceAudioChunkIn` | `IntakeAck` | required `case_id`, `call_id`, `audio`; optional provider `transcript_text` |
| `GET /health` | none | JSON | returns `{ "status": "ok", "layer": "layer0-input" }` |

`IntakeAck` has `schema_version`, `case_id`, `input_id`, `channel`, `modalities`, `accepted`, and `dispatch_queue`.

### Planned — not implemented

- `GET /cases`
- `GET /case/{case_id}`
- `POST /case/{case_id}/confirm`
- Evidence, SVI, service, support, dashboard, and audit endpoints

Do not expose or depend on these planned endpoints until the owning team implements and tests them.

## 10. Data validation and privacy

- New case IDs are generated as `CASE-...`; supplied IDs must be non-empty and start with `CASE-` in the current implementation.
- Every `InputEnvelope` has a generated `input_id` and UTC `timestamp`.
- `language_hint` is optional and is not language detection.
- At least one modality is required. Declared `text`, `audio`, and `structured_data` each need matching content.
- Text limit: 8,000 characters. Audio duration ceiling: 3,600 seconds. Current supported audio MIME types are `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp3`, `audio/ogg`, `audio/webm`, and `audio/L16`.
- `audio_ref_id` is required for audio; raw audio bytes do not enter the envelope.
- `consent` records intake declaration only. Later policy enforcement must be proposed separately.
- Metadata must be non-sensitive operational information. Do not place raw PII, raw complaint text, transcripts, or audio bytes in logs or metadata.

## 11. Error handling

Current Layer 0 deterministic validation errors return HTTP 400:

```json
{"error": {"code": "MISSING_AUDIO_REF_ID", "message": "...", "field": "audio.audio_ref_id"}}
```

FastAPI request-shape errors currently use its normal HTTP 422 response. Unexpected errors return HTTP 500 with the generic `INTERNAL_ERROR` body and no stack trace.

Future modules should return a structured error object with stable `code`, safe `message`, optional `field`, and optional safe correlation ID. Candidate categories include `UNSUPPORTED_LANGUAGE`, `STT_FAILED`, `MODEL_FAILED`, `RAG_FAILED`, `TIMEOUT`, and `EXTERNAL_SERVICE_FAILED`. Do not return raw provider exceptions or sensitive inputs.

## 12. Integration rules

1. Do not casually rename shared schemas, files, fields, endpoints, or public adapters.
2. Update this document before changing an API response or cross-module schema.
3. Internal refactors are allowed when the external contract remains stable.
4. Do not import or call another member's internal implementation directly; depend on schemas/interfaces or an agreed adapter.
5. Use mocks/fakes when an upstream/downstream module is incomplete.
6. Add integration tests at each schema boundary.
7. Keep changes within the responsible module; coordinate any cross-owner edit.
8. Never add Layer 1–5 assessment/decision behavior to Layer 0.

## 13. Git rules

- `main` is the protected stable integration branch.
- Use feature branches: `feature/input`, `feature/fusion`, `feature/svi`, `feature/rag`, `feature/orchestration`, `feature/dashboard`.
- No direct pushes to `main`; open a pull request and get review before merge.
- Contract changes require team agreement and an update to `CONTRACTS.md` in the same PR.
- Do not modify another member's module without notifying the owner.
- Rebase/resolve conflicts locally, run relevant tests, and keep commits focused.

## 14. Current status

| Area | Status | Verified scope |
| --- | --- | --- |
| Layer 0 — Input / Ingestion | **CURRENT IMPLEMENTATION** | FastAPI endpoints, validation, envelope, deterministic routing, in-memory case/session/dispatch adapters, privacy-aware logging, 16 focused tests passing in the supplied source |
| Layer 1 — Fusion / Privacy | **PLANNED** | No implementation added |
| Layer 2 — SVI / Risk | **PLANNED** | No implementation added |
| Layer 3 | **PLANNED / architecture placeholder** | No separate module defined yet; agree scope before creating one |
| Layer 4 — Service | **PLANNED** | No implementation added |
| Layer 4 — Support | **PLANNED** | No implementation added |
| Layer 5 — Dashboard / API integration | **PLANNED** | No implementation added |

### Decisions the team should discuss

1. The conceptual brief uses `chat`; existing working Layer 0 uses the canonical value `chatbot`. Preserve `chatbot` unless the whole team approves a versioned migration.
2. `POST /voice/incoming` is a session-start event, not evidence intake. It creates no `InputEnvelope`; real voice evidence first arrives through `/voice/chunk`. Decide whether future case/session persistence needs to retain call-start language or consent details.
3. Decide the exact persistence/queue adapter and case-management API only when the respective owners begin implementation. Current in-memory adapters are development-only.

## 15. Contract change procedure

```text
Propose -> Discuss -> Update CONTRACTS.md -> Implement -> Test -> Merge
```

Breaking changes must never be silently introduced. A breaking change needs team agreement, a migration/compatibility plan, contract versioning where needed, focused tests, and a reviewed pull request.
