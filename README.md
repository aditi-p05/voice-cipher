# Team Voice Cipher — NHAA 14566

SIH 2026, Problem Statement 26093: AI-assisted real-time stress and trauma assessment support for victims/complainants accessing NHAA (14566) and its integrated portal.

This repository begins with the working Layer 0 ingestion implementation. It accepts voice-call events and audio references, chatbot messages, and complaint-portal submissions, normalizes them into an `InputEnvelope`, and dispatches the envelope through a development in-memory dispatcher.

The repository deliberately does **not** implement fusion, PII redaction, SVI/risk, RAG, orchestration, dashboard, or operator decision logic yet. Those teams integrate through the stable interfaces in [CONTRACTS.md](CONTRACTS.md).

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the current Layer 0 API.

## Current layout

- `backend/` — working Layer 0 FastAPI implementation, owned by Member 1 until integration ownership is agreed.
- `tests/` — Layer 0 focused tests.
- `services/` — reserved module ownership locations; intentionally contains no later-layer implementation.
- `frontend/react-app/` — reserved for Member 6's frontend.
- `data/` — local sample/SOP locations. Never commit real complainant data or recordings.
- `docs/` — architecture notes.

See [CONTRACTS.md](CONTRACTS.md) before changing any endpoint, shared schema, enum, or inter-module interface.
