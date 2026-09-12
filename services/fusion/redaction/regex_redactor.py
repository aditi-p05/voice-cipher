"""
Layer 1 (Fusion / Privacy) — default PII redactor.

Deliberately dependency-light (stdlib `re` only) so `pytest -q` never
needs a model download. This is a deterministic, explainable, best-effort
redactor suitable for an SIH demo — not a substitute for a trained NER
model in production. See the "Known limitations" note in the integration
report: name detection in particular is heuristic (title/trigger-phrase
based) and will miss names with no surrounding cue, and can occasionally
over-redact a capitalized non-name phrase. `PresidioPiiRedactor` (see
presidio_redactor.py) can replace this behind the same `PiiRedactor`
interface without touching fusion_service.py.

Never logs or stores the raw input text; only returns the redacted
version plus non-sensitive metadata about *where* something was redacted.
"""

from __future__ import annotations

import re

from services.fusion.redaction.base import PiiFinding, PiiRedactor, RedactionResult

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Phone numbers: optional country code, 10-13 digits, optionally grouped
# with spaces/dashes/dots. Deliberately broad for a demo redactor.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d(?!\d)"
)

# Common government/financial identifier shapes (India-focused, since this
# is an NHAA/SIH context) — Aadhaar-like 12-digit, PAN-like 10-char.
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

# Address-ish cues: house/plot/flat numbers plus a street-type keyword
# nearby. Best-effort; full postal-address extraction needs a real NER
# model and is out of scope for this lightweight default.
_ADDRESS_RE = re.compile(
    r"\b(?:house|flat|plot|block)\s*(?:no\.?|number)?\s*[:\-]?\s*\S+"
    r"(?:[,\s]+[A-Za-z0-9.'-]+){0,6}"
    r"(?:street|st\.|road|rd\.|lane|nagar|colony|sector|avenue)\b[^.,;\n]*",
    re.IGNORECASE,
)

# Name cues: honorific, or a first-person self-introduction phrase,
# followed by 1-3 capitalized words. Heuristic only — see module
# docstring. NOTE: the trigger words use a scoped inline (?i:) flag
# rather than a global re.IGNORECASE, deliberately — a global IGNORECASE
# would also fold [A-Z] to match lowercase, silently turning this into
# "redact any word after 'I am'" and swallowing ordinary sentences.
_NAME_HONORIFIC_RE = re.compile(
    r"\b(?:(?i:Mr|Mrs|Ms|Dr|Shri|Smt))\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
)
_NAME_SELF_INTRO_RE = re.compile(
    r"\b(?:(?i:my name is|i am|this is|i'm))\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}"
)


class RegexPiiRedactor(PiiRedactor):
    # Order matters: more specific patterns first so a later, broader
    # pattern (e.g. phone) doesn't clobber an already-redacted span.
    _PATTERN_ORDER: tuple[tuple[str, re.Pattern], ...] = (
        ("EMAIL_ADDRESS", _EMAIL_RE),
        ("AADHAAR_NUMBER", _AADHAAR_RE),
        ("PAN_NUMBER", _PAN_RE),
        ("ADDRESS", _ADDRESS_RE),
        ("PERSON", _NAME_HONORIFIC_RE),
        ("PERSON", _NAME_SELF_INTRO_RE),
        ("PHONE_NUMBER", _PHONE_RE),
    )

    def redact(self, text: str, language: str | None = None) -> RedactionResult:
        if not text:
            return RedactionResult(redacted_text=text or "", findings=[])

        working = text
        findings: list[PiiFinding] = []

        for entity_type, pattern in self._PATTERN_ORDER:
            working, matches = self._apply(working, pattern, entity_type)
            findings.extend(matches)

        return RedactionResult(redacted_text=working, findings=findings)

    @staticmethod
    def _apply(
        text: str, pattern: re.Pattern, entity_type: str
    ) -> tuple[str, list[PiiFinding]]:
        findings: list[PiiFinding] = []
        placeholder = f"[REDACTED_{entity_type}]"

        def _sub(match: re.Match) -> str:
            findings.append(PiiFinding(entity_type=entity_type, start=match.start(), end=match.end()))
            return placeholder

        new_text = pattern.sub(_sub, text)
        return new_text, findings
