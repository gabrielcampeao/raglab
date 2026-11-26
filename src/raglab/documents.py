from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from raglab.models import Document

SUPPORTED_SUFFIXES = (".md", ".txt")


def load_directory(
    path: str | Path, suffixes: Iterable[str] = SUPPORTED_SUFFIXES
) -> list[Document]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    allowed = {suffix.lower() for suffix in suffixes}
    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in allowed:
            continue
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative = file.relative_to(root).as_posix()
        documents.append(Document(id=relative, text=text, metadata={"source": relative}))
    return documents
