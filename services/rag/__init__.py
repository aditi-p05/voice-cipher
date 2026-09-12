"""Traceable, local SOP retrieval for operator-confirmed recommendations."""

from .service import generate_recommendation
from .store import InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "generate_recommendation"]
