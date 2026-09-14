"""
Layer 4B (Orchestration) — soft timeout helper for node calls.

Any stage (fusion, SVI, RAG, support) may eventually wrap a slow external
model or provider. A hung call must not hang the whole pipeline; it must
degrade into a structured TIMEOUT error like any other stage failure.

Implemented with a thread pool rather than signals so it stays safe to
call from any context (e.g. inside FastAPI/uvicorn workers).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Callable, TypeVar

T = TypeVar("T")


class NodeTimeoutError(Exception):
    """Raised when a wrapped call exceeds its time budget."""


def run_with_timeout(fn: Callable[[], T], *, seconds: float) -> T:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except FutureTimeoutError as exc:
            raise NodeTimeoutError(f"Call exceeded {seconds}s timeout.") from exc
