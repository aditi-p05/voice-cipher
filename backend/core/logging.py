"""
Layer 0 — minimal structured logging.

Hard rule: never accept raw text/audio content as a loggable value. The
event schema below only allows non-content fields, so accidental
`log_event("input_received", text=raw_text)` calls fail loudly instead of
leaking sensitive content into logs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("nhaa.layer0")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Fields that must never be passed to log_event, enforced defensively even
# though callers shouldn't have raw content available at this layer.
_FORBIDDEN_FIELDS = {"text", "body", "raw_text", "audio_bytes", "transcript", "complaint_text"}


def log_event(event: str, **fields) -> None:
    forbidden_present = _FORBIDDEN_FIELDS.intersection(fields.keys())
    if forbidden_present:
        raise ValueError(
            f"Refusing to log forbidden sensitive field(s): {sorted(forbidden_present)}"
        )

    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    logger.info(json.dumps(record, default=str))
