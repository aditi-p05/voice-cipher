"""
Layer 1 (Fusion / Privacy) — optional Presidio-backed PII redactor.

Not used by default and not imported anywhere else in this package, so
`pytest -q` never requires installing `presidio-analyzer`/`presidio-
anonymizer` or downloading a spaCy model. Instantiate this explicitly
(e.g. in a production wiring/config layer) to upgrade redaction quality
beyond the regex default, without changing any fusion_service call site —
both implement the same `PiiRedactor` interface.
"""

from __future__ import annotations

from services.fusion.errors import MODEL_FAILED, FusionError
from services.fusion.redaction.base import PiiFinding, PiiRedactor, RedactionResult


class PresidioPiiRedactor(PiiRedactor):
    def __init__(self, language: str = "en"):
        self._language = language
        self._analyzer = None
        self._anonymizer = None

    def _load(self):
        if self._analyzer is not None:
            return
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
            from presidio_anonymizer import AnonymizerEngine  # type: ignore
        except ImportError as exc:
            raise FusionError(
                MODEL_FAILED,
                "Presidio redactor requested but 'presidio-analyzer'/"
                "'presidio-anonymizer' are not installed in this environment.",
                details={"provider": "presidio"},
            ) from exc
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

    def redact(self, text: str, language: str | None = None) -> RedactionResult:
        if not text:
            return RedactionResult(redacted_text=text or "", findings=[])

        self._load()
        lang = language or self._language
        try:
            results = self._analyzer.analyze(text=text, language=lang)
            anonymized = self._anonymizer.anonymize(text=text, analyzer_results=results)
        except Exception as exc:  # never leak raw provider internals/text upward
            raise FusionError(
                MODEL_FAILED,
                "PII redaction model failed.",
                details={"provider": "presidio"},
            ) from exc

        findings = [
            PiiFinding(entity_type=r.entity_type, start=r.start, end=r.end) for r in results
        ]
        return RedactionResult(redacted_text=anonymized.text, findings=findings)
