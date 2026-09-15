# Voice Cipher — NHAA Operational Dashboard

Desktop-focused React + Tailwind dashboard for the Voice Cipher case workflow.

## Run

```powershell
npm install --cache .npm-cache
npm run dev
```

The dashboard automatically uses `http://127.0.0.1:8000` (or `VITE_API_BASE_URL`) and integrates with:

- `GET /cases` and `GET /case/{case_id}` for the active case display
- `POST /case/{case_id}/confirm` for human `CONFIRM` / `OVERRIDE` audit actions

When the API has no live cases or is offline, it uses a clearly tagged local demo case so the UI remains reviewable. Raw audio or personally identifying data is never rendered; transcript redaction is visually preserved with `REDACTED` pills.

The project intentionally keeps the decision action as a human operator confirmation. It does not present automated emergency, legal, medical, or policing actions as executable.
