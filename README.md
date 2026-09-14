# Team Voice Cipher — NHAA 14566

SIH 2026, Problem Statement 26093: AI-assisted real-time stress and trauma assessment support for victims/complainants accessing NHAA (14566) and its integrated portal.

This repository contains the working Layer 0 ingestion implementation, Layer 1 Fusion/Privacy, Layer 4A Service/RAG, and Layer 4B Orchestration/Support. It accepts voice-call events and audio references, chatbot messages, and complaint-portal submissions; normalizes them into an `InputEnvelope`; fuses them into a redacted `EvidenceBundle`; and (via `backend.orchestration.run.run_pipeline`) runs the full pipeline through to a risk-gated support decision and a procedural service recommendation, both requiring human operator confirmation.

SVI/Risk (Layer 2, Member 3) is now implemented (`services/svi/`) as of the `feature/svi` merge. This branch (`feature/orchestration`, Member 5) wires it into the real pipeline via `backend/orchestration/adapters/svi_adapter.py`, which prefers Member 3's real `services.svi.svi_service.calculate_svi` over the earlier documented placeholder. The dashboard/operator-decision API (Layer 5, Member 6) is still **not yet implemented** (`frontend/react-app/` is still reserved). See CONTRACTS.md §14 for current per-layer status.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt   # includes langgraph for orchestration
pytest -q
uvicorn backend.main:app --reload
python scripts/run_demo.py        # run a few synthetic cases through the full pipeline
```

Open `http://127.0.0.1:8000/docs` to use the current Layer 0 API.

## Operator dashboard and case API

Each accepted intake is processed through the available pipeline and added to an in-memory, redacted operator case read model. `GET /cases`, `GET /case/{case_id}`, and `POST /case/{case_id}/confirm` provide the dashboard API. The confirmation route accepts `operator_id`, `action` (`CONFIRM` or `OVERRIDE`), optional `final_decision`, and an override `reason` (required for overrides). Every action is append-only in the case audit history; no automated action is executed.

Run the dashboard after starting the backend:

```powershell
cd frontend/react-app
npm install
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000" # optional
npm run dev
```

The development case read model is process-local and intentionally replaceable by durable persistence later.

## Current layout

- `backend/` — Layer 0 FastAPI implementation (Member 1) plus `backend/orchestration/` — the LangGraph pipeline, Support Engine glue, voice-session accumulation, and audit consumption (Member 5).
- `services/fusion/` — Layer 1 Fusion/Privacy (Member 2). `services/rag/` — Layer 4A Service/RAG (Member 4). `services/support/` — Layer 4B risk-gated Support Engine (Member 5). `services/svi/` — reserved for Layer 2 SVI/Risk (Member 3), not yet implemented.
- `scripts/run_demo.py` — runs a handful of synthetic cases through the real pipeline end-to-end and prints a summary.
- `docs/proposals/` — proposed CONTRACTS.md additions awaiting team discussion (not yet part of the contract).
- `tests/` — focused tests per layer/module.
- `frontend/react-app/` — reserved for Member 6's frontend.
- `data/` — local sample/SOP locations. Never commit real complainant data or recordings.

See [CONTRACTS.md](CONTRACTS.md) before changing any endpoint, shared schema, enum, or inter-module interface.
