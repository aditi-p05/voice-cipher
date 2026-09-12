"""Internal value objects; public results are shared-contract dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SopChunk:
    """A retrievable non-sensitive SOP passage with a stable citation."""

    text: str
    metadata: Mapping[str, str]

    @property
    def citation(self) -> str:
        return (
            f"{self.metadata['sop_id']} v{self.metadata['version']} | "
            f"{self.metadata['document_name']} | {self.metadata['section']} | "
            f"{self.metadata['chunk_id']}"
        )


@dataclass(frozen=True)
class RetrievalMatch:
    chunk: SopChunk
    score: float
