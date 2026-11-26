from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from raglab.models import Document

SUPPORTED_SUFFIXES = (".md", ".txt")
MANIFEST_NAME = "metadata.json"


def _load_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest_file = root / MANIFEST_NAME
    if not manifest_file.is_file():
        return {}
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{MANIFEST_NAME} must contain a JSON object")
    parsed: dict[str, dict[str, str]] = {}
    for doc_id, fields in raw.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Metadata entry for '{doc_id}' must be a JSON object")
        parsed[str(doc_id)] = {str(key): str(value) for key, value in fields.items()}
    return parsed


def load_directory(
    path: str | Path, suffixes: Iterable[str] = SUPPORTED_SUFFIXES
) -> list[Document]:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    allowed = {suffix.lower() for suffix in suffixes}
    manifest = _load_manifest(root)
    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in allowed:
            continue
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative = file.relative_to(root).as_posix()
        metadata = {**manifest.get(relative, {}), "source": relative}
        documents.append(Document(id=relative, text=text, metadata=metadata))
    return documents
