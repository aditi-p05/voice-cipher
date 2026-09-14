from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any


class InMemoryCaseReadStore:
    """Replaceable development read model; stores only redacted pipeline output."""
    def __init__(self) -> None:
        self._cases: dict[str, dict[str, Any]] = {}
        self._actions: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()

    def upsert_pipeline_view(self, view: dict[str, Any]) -> None:
        with self._lock:
            case_id = view["case_id"]
            existing = self._cases.get(case_id, {})
            self._cases[case_id] = {**existing, **deepcopy(view), "updated_at": datetime.now(timezone.utc).isoformat()}

    def get(self, case_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._cases.get(case_id)
            return deepcopy(value) if value else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(v) for v in sorted(self._cases.values(), key=lambda c: c.get("updated_at", ""), reverse=True)]

    def append_action(self, action: dict[str, Any]) -> None:
        with self._lock:
            self._actions.setdefault(action["case_id"], []).append(deepcopy(action))

    def actions(self, case_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._actions.get(case_id, []))
