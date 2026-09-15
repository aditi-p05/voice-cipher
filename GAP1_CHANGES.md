# Gap 1 — what I added and why

## The real deliverable (what you asked for)
`backend/orchestration/adapters/fusion_adapter.py`
- Calls **Gemini Audio Understanding** (`client.interactions.create` with
  a structured `response_format`) on the audio at
  `AudioReference.storage_uri`.
- Gets back verbatim transcript, per-segment timestamps, language, and
  emotion/tone cues (per-segment + overall valence/arousal).
- Runs a **regex PII redaction fallback** on the transcript.
- Runs **rule-based explicit danger/threat marker extraction** on the
  *redacted* text — separate from Gemini's tone read, on purpose (tone
  alone never becomes a Marker, only explicit phrases do).
- Builds the `EvidenceBundle` your `evidence_merge.py`/`run.py` already
  expect.
- For chat/portal text (no audio), skips Gemini entirely and runs the
  same redaction + marker steps directly on the text.

Tested offline (mocked the Gemini client so no real API call was made
from here) — both the text-only path and the audio path produce the
expected transcript/markers/sentiment. You'll still want to run it
against a real audio file with your real `GEMINI_API_KEY` before demo
day, since I can't reach `generativelanguage.googleapis.com` from here.

## Two gaps I hit that weren't part of your ask, but blocked everything
1. **`services/fusion/evidence_bundle.py` didn't exist in this zip** —
   `EvidenceBundle`/`Marker`/`Transcript`/`Sentiment` are your Member
   2's contract and every module (`evidence_merge.py`, `run.py`,
   `state.py`) already imports them. I wrote a minimal version matching
   exactly how those files already use it (see the big comment at the
   top of that file). **Swap it for Member 2's real file the moment you
   have it** — the field names should already line up.
2. **`backend/orchestration/adapters/{svi,rag,support}_adapter.py`
   didn't exist either** — `nodes.py` imports all three unconditionally,
   so nothing in the graph could even be imported. These are now all
   **Gemini-backed** too (see below), not mocks.

## What's still a placeholder, on purpose
- PII redaction is regex-only (phone/Aadhaar-shaped/email/PIN code).
  It won't catch names or addresses. Swap `redact_pii()`'s body for the
  real Presidio + spaCy pipeline you're retaining from Detox.ai when
  you're ready to pull that dependency back in.
- The danger/threat phrase list in `_DANGER_PATTERNS` is a starting set
  (English + a few Hindi/Hinglish patterns). Worth a pass with whoever
  on your team knows the PoA Act phrasing best.
- `voice_features` (pitch/jitter/prosody) is left `None` here — that's
  your separate `/analyze-voice` engine's job, not this adapter's.

## To run it for real
```
pip install google-genai fastapi pydantic python-dotenv langgraph
# .env: GEMINI_API_KEY=...   (GEMINI_MODEL defaults to gemini-2.5-flash)
```
Then point an `AudioReference.storage_uri` at a local `.wav`/`.mp3` file
and call `backend.orchestration.run.run_pipeline(envelope)` (or go
through `/voice/incoming` + `/voice/chunk` if you're testing via the
API). If the model errors are about `interactions` API not existing.on
your installed SDK version, run `pip install --upgrade google-genai`
first — that endpoint is fairly new.


---

# Update — all four stages are now Gemini-backed

You wanted the Gemini key to cover the whole pipeline, not just
transcription. Done. `backend/orchestration/adapters/gemini_client.py`
is the shared call helper; each stage has its own prompt + structured
output schema.

| Stage | File | What Gemini does |
|---|---|---|
| Fusion | `fusion_adapter.py` | audio → transcript, timestamps, language, tone |
| SVI | `svi_adapter.py` | evidence → svi_score, risk_tier, rationale, key_factors |
| Service | `rag_adapter.py` | evidence + risk → recommended service routes + rationale |
| Support | `support_adapter.py` | evidence + risk → operator notes, next questions, cautions |

There are **no mocks left in the default path.** Every stage hits the
real API using `GEMINI_API_KEY` from your `.env`.

## Three things I did NOT let the model decide

These are deliberate, and I'd push back if anyone wants them removed:

1. **Safety floor on the risk tier** (`svi_adapter._apply_safety_floor`).
   If the rule-based pass found a `critical` marker, the tier can't come
   out below Critical; a `high` marker floors it at High. The floor only
   moves risk **up**, never down. Reason: one bad model response
   downgrading an explicit same-day death threat is a failure a helpline
   cannot absorb, and it's the same dilution your `evidence_merge.py`
   already refuses to allow across chunks. When it fires it's visible in
   the output (`safety_floor_applied: true`, `model_risk_tier`), not silent.
   Verified in testing: model said "Moderate", output was "Critical".

2. **Service routes are allow-listed** (`rag_adapter.ROUTES`). Anything
   outside the five real tracks gets dropped rather than reaching the
   dashboard, so the model can't invent a service the helpline doesn't run.
   Verified: a junk route in the response was filtered out.

3. **`requires_operator_confirmation` and `operator_alert` are computed
   in code from the risk tier**, never read from the model. The model
   must not be able to talk the system out of a human check.

Also: the support prompt explicitly forbids drafting caller-facing
script, diagnosing the caller, or implying help is already on the way.
It produces notes *for the operator* only.

## Failure behaviour (tested)
Simulated a Gemini outage: fusion fails → `evidence_available: false`,
`svi_result: null`, `requires_operator_confirmation: false`, and one
structured `FUSION_FAILED` error. No fabricated evidence, no invented
score, no raw exception. Matches the philosophy in `nodes.py`.

## New config
`NHAA_SVI_TIMEOUT_SECONDS` (default 10s) added to `config.py` — SVI used
to be local arithmetic with no timeout; it's a network call now.

## Cost/latency note worth knowing before demo day
This is now **4 Gemini calls per voice chunk**, and
`orchestration/voice_session.py` re-runs SVI/Service/Support on *every*
chunk arrival by design. A 6-chunk call = ~19 API calls. That module's
docstring says to revisit batching "with real latency numbers in hand"
once a real provider is in play — you now have a real provider, so it's
worth timing a full call before the demo and deciding whether the
non-critical Support branch should run less often.


---

# Update — API layer was never importable; now fixed

While checking a specific voice-session-state question, found the
FastAPI app itself couldn't be imported at all:

    ModuleNotFoundError: No module named 'backend.routing'

Two more modules were missing (Layer 0 / Member 1's territory, separate
from anything voice/Gemini-related):

- `backend/routing/modality_router.py` — `RoutingResult` + `route()`.
  `intake_service.py` already called this unconditionally.
- `backend/transport/dispatcher.py` — `InputDispatcher` (ABC),
  `DispatchResult`, `InMemoryDispatcher`. `intake_service.py` and
  `dependencies.py` both already depended on it.

Added minimal stubs for both, clearly marked as placeholders for
Member 1's real implementation. `RoutingResult` is stored on
`IntakeOutcome` but nothing else in this codebase reads it today, so
the stub's exact routing rules are low-stakes to replace later.

**Verified with real HTTP requests** (FastAPI TestClient, Gemini
mocked): `/health`, `/voice/incoming`, two sequential `/voice/chunk`
calls on the same `call_id`, and `/cases` — confirmed the second
(neutral) chunk didn't dilute the first chunk's Critical tier. This is
the first time this pipeline was exercised through the actual API
rather than by calling `run_pipeline()` directly, so it's also the
first real confirmation that Layer 0 -> Layer 4B -> the four Gemini
adapters are correctly wired together end-to-end.

Every backend module now imports; `backend.main:app` is genuinely
runnable with `uvicorn backend.main:app --reload` given a real
`GEMINI_API_KEY`.
