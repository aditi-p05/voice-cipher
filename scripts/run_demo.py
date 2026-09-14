#!/usr/bin/env python3
"""
Layer 4B (Orchestration) — demo CLI.

Runs a handful of representative synthetic cases through the REAL
pipeline (real fusion, real RAG, the risk-gated Support Engine, and the
explicitly-flagged mock SVI provider standing in for Member 3's not-yet-
implemented Layer 2) and prints a compact summary of each.

Usage:
    python scripts/run_demo.py

No real complainant data is used or required -- every case below is
synthetic text written for this demo only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingestion.input_envelope import InputEnvelope, TextContent  # noqa: E402
from backend.models.enums import Channel, Modality  # noqa: E402
from backend.orchestration.run import run_pipeline  # noqa: E402

DEMO_CASES = [
    ("mild / routine inquiry", "I just wanted to ask about the process for filing a complaint."),
    ("distress disclosure", "I am scared and I feel very alone since this happened."),
    ("threat disclosure", "He threatened to hurt me if I told anyone, and I am terrified."),
    ("acute crisis disclosure", "I want to end my life, he had a knife and he is outside right now."),
]


def summarize(label: str, text: str) -> None:
    envelope = InputEnvelope(
        case_id=f"CASE-DEMO-{abs(hash(text)) % 10000:04d}",
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        text=TextContent(body=text),
    )
    case = run_pipeline(envelope)

    print(f"\n=== {label} ===")
    print(f"case_id:            {case['case_id']}")
    print(f"risk_tier:          {case['risk_tier']}")
    svi = case["svi_result"] or {}
    print(f"svi_score:          {svi.get('svi_score')} (rule_floor={svi.get('rule_floor')}, ml_score={svi.get('ml_score')})")
    markers = (case["evidence_bundle"] or {}).get("markers", [])
    print(f"markers:            {[m['type'] for m in markers] or '(none)'}")
    support = case["support_decision"] or {}
    print(f"support:            enabled={support.get('support_enabled')} mode={support.get('mode')}")
    service = case["service_recommendation"] or {}
    recs = service.get("recommendations", [])
    print(f"service_recs:       {len(recs)} recommendation(s), requires_confirmation={case['requires_operator_confirmation']}")
    if case["errors"]:
        print(f"errors:             {case['errors']}")


def main() -> None:
    print("NHAA Layer 4B demo -- synthetic cases only, no real complainant data.")
    print(
        "svi_result below comes from Member 3's real "
        "services.svi.svi_service.calculate_svi (feature/svi merged into main).\n"
    )
    for label, text in DEMO_CASES:
        summarize(label, text)

    print("\n--- Full JSON for the CRITICAL case (for inspection) ---")
    envelope = InputEnvelope(
        case_id="CASE-DEMO-FULL",
        channel=Channel.CHATBOT,
        modalities=[Modality.TEXT],
        text=TextContent(body=DEMO_CASES[-1][1]),
    )
    print(json.dumps(run_pipeline(envelope), indent=2, default=str))


if __name__ == "__main__":
    main()
