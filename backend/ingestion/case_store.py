"""
Layer 0 — smallest possible case abstraction.

This is NOT a case-management database. It only provides enough of an
interface so that:
  * a new case_id can be minted,
  * an existing case_id can be validated as "known",
  * multiple inputs (chat messages, voice segments, portal text) can be
    associated with the same case_id.

A real persistence layer (SQL/NoSQL) can implement CaseStore later without
touching ingestion/API code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.ingestion.input_envelope import new_case_id


class CaseStore(ABC):
    @abstractmethod
    def create_case(self) -> str:
        """Mint and register a brand-new case_id."""
        raise NotImplementedError

    @abstractmethod
    def register_case_id(self, case_id: str) -> None:
        """Register a caller-supplied case_id that is not yet known."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, case_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def register_input(self, case_id: str, input_id: str) -> None:
        """Associate an input_id with an existing/new case_id."""
        raise NotImplementedError

    @abstractmethod
    def input_ids_for_case(self, case_id: str) -> list[str]:
        raise NotImplementedError


class InMemoryCaseStore(CaseStore):
    """Process-local case registry. Fine for dev/tests; not durable."""

    def __init__(self):
        self._cases: dict[str, list[str]] = {}

    def create_case(self) -> str:
        case_id = new_case_id()
        self._cases[case_id] = []
        return case_id

    def register_case_id(self, case_id: str) -> None:
        self._cases.setdefault(case_id, [])

    def exists(self, case_id: str) -> bool:
        return case_id in self._cases

    def register_input(self, case_id: str, input_id: str) -> None:
        self._cases.setdefault(case_id, []).append(input_id)

    def input_ids_for_case(self, case_id: str) -> list[str]:
        return list(self._cases.get(case_id, []))
