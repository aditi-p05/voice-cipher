"""Safe loading and section-aware chunking of approved local SOP Markdown."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .models import SopChunk

_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_META = re.compile(r"^\s*<!--\s*(SOP_ID|VERSION|SOURCE)\s*:\s*(.+?)\s*-->\s*$", re.I)


def load_sop_documents(directory: str | Path) -> list[tuple[Path, str]]:
    """Load only local Markdown SOP files; never accept case data as SOP input."""
    root = Path(directory)
    if not root.exists():
        return []
    return [(path, path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.md"))]


def chunk_sop_document(path: str | Path, content: str, chunk_size: int = 700) -> list[SopChunk]:
    """Split a Markdown SOP by heading, retaining source/version metadata."""
    path = Path(path)
    values = {"SOP_ID": path.stem.upper(), "VERSION": "unversioned", "SOURCE": str(path)}
    body: list[str] = []
    section = "Introduction"
    chunks: list[SopChunk] = []

    def emit(lines: Iterable[str]) -> None:
        text = "\n".join(lines).strip()
        if not text:
            return
        # Paragraph boundaries are preferable; hard split only for unusually long sections.
        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        for part in parts:
            index = len(chunks) + 1
            chunks.append(SopChunk(part, {
                "document_name": path.name,
                "sop_id": values["SOP_ID"],
                "section": section,
                "version": values["VERSION"],
                "source": values["SOURCE"],
                "chunk_id": f"{values['SOP_ID']}-{index:03d}",
            }))

    for line in content.splitlines():
        meta = _META.match(line)
        if meta:
            values[meta.group(1).upper()] = meta.group(2)
            continue
        header = _HEADER.match(line)
        if header:
            emit(body)
            body = []
            section = header.group(2)
        else:
            body.append(line)
    emit(body)
    return chunks
