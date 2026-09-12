"""Dependency-free local vector store for the SIH demo."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

from .models import RetrievalMatch, SopChunk

_TOKENS = re.compile(r"[a-z0-9_]+", re.I)


class VectorStore(Protocol):
    def add(self, chunks: Sequence[SopChunk]) -> None: ...
    def search(self, query: str, top_k: int = 3) -> list[RetrievalMatch]: ...


def embed(text: str, dimensions: int = 256) -> list[float]:
    """Stable hashing embedding; local, reproducible, and free of API keys."""
    vector = [0.0] * dimensions
    for token in _TOKENS.findall(text.lower()):
        slot = int(hashlib.sha256(token.encode()).hexdigest(), 16) % dimensions
        vector[slot] += 1.0
    length = math.sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length else vector


class InMemoryVectorStore:
    """Small FAISS-like cosine index retained only in process memory."""
    def __init__(self) -> None:
        self._entries: list[tuple[SopChunk, list[float]]] = []

    def add(self, chunks: Sequence[SopChunk]) -> None:
        # Section titles carry the procedural/risk label for otherwise terse action lists.
        self._entries.extend(
            (chunk, embed(f"{chunk.metadata.get('section', '')} " * 5 + chunk.text)) for chunk in chunks
        )

    def search(self, query: str, top_k: int = 3) -> list[RetrievalMatch]:
        if not query.strip():
            return []
        needle = embed(query)
        matches = [RetrievalMatch(chunk, sum(a * b for a, b in zip(needle, vector)))
                   for chunk, vector in self._entries]
        return [match for match in sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]
                if match.score > 0]
