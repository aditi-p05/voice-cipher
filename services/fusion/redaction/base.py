"""
Layer 1 (Fusion / Privacy) — pluggable PII redaction abstraction.

Nothing downstream of `build_evidence_bundle` may see un-redacted text
(CONTRACTS.md 6.2: "EvidenceBundle must contain REDACTED text only").
`PiiRedactor` is the seam so the concrete engine (regex-based default,
Presidio, a cloud DLP API, ...) can change without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PiiFinding:
    """Metadata about one redacted span. Never contains the raw value."""

    entity_type: str  # e.g. "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"
    start: int
    end: int


@dataclass(frozen=True)
class RedactionResult:
    redacted_text: str
    findings: list[PiiFinding] = field(default_factory=list)

    @property
    def any_pii_found(self) -> bool:
        return len(self.findings) > 0


class PiiRedactor(ABC):
    """Common interface every redaction backend must implement."""

    @abstractmethod
    def redact(self, text: str, language: str | None = None) -> RedactionResult:
        """
        Return `text` with sensitive spans replaced by placeholders (e.g.
        "[REDACTED_PHONE_NUMBER]"). Must never raise on merely "no PII
        found" — that is simply a `RedactionResult` with no findings.
        Implementations must never log or persist the raw input text.
        """
        raise NotImplementedError
