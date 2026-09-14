"""
Layer 4B (Orchestration) — audit consumption of OperatorAction.

CONTRACTS.md section 2 lists orchestration as a CONSUMER of
`OperatorAction` (produced by Member 6 / the human operator), alongside
audit. No audit persistence layer exists anywhere in this repo yet --
that's a cross-cutting decision (not owned by any single member) and is
deliberately not invented here. This module gives Member 6 something
concrete to call today: a validated receiving shape plus a safe,
structured log record, matching the pattern every other layer uses
(backend/core/logging.py's `log_event`, which already refuses to log
raw content fields).

Swap `record_operator_action`'s logging call for a real persistent audit
store once the team agrees on one; the validation contract here should
not need to change when that happens.
"""

from __future__ import annotations

from typing import Any, Mapping

from backend.core.logging import log_event

_VALID_ACTIONS = {"CONFIRM", "OVERRIDE"}
_REQUIRED_FIELDS = ("schema_version", "case_id", "operator_id", "action", "timestamp")


class InvalidOperatorAction(ValueError):
    """Raised for a structurally invalid OperatorAction. Never silently accepted."""


def record_operator_action(action: Mapping[str, Any]) -> None:
    """
    Validate and audit-log one OperatorAction (CONTRACTS.md 6.5).

    Deliberately strict: a missing required field or an invalid `action`
    value raises rather than being logged as if it were valid -- an audit
    trail with silently-accepted garbage is worse than no audit trail.
    """
    for field_name in _REQUIRED_FIELDS:
        if field_name not in action:
            raise InvalidOperatorAction(f"OperatorAction missing required field: {field_name}")

    if action["action"] not in _VALID_ACTIONS:
        raise InvalidOperatorAction("OperatorAction.action must be CONFIRM or OVERRIDE")

    log_event(
        "operator_action_recorded",
        case_id=action["case_id"],
        operator_id=action["operator_id"],
        action=action["action"],
        had_override_reason=bool(action.get("reason")),
    )
