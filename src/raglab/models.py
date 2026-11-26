from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    document_id: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", self.document_id)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
