from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from raglab.models import Chunk, SearchResult
from raglab.vectormath import Vector, cosine_similarity


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[Vector] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimensions(self) -> int | None:
        return len(self._vectors[0]) if self._vectors else None

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def add(self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]) -> None:
        new_chunks = list(chunks)
        new_vectors = [[float(value) for value in vector] for vector in vectors]
        if len(new_chunks) != len(new_vectors):
            raise ValueError("Each chunk needs exactly one vector")
        if not new_vectors:
            return
        expected = self.dimensions if self.dimensions is not None else len(new_vectors[0])
        if any(len(vector) != expected for vector in new_vectors):
            raise ValueError(f"All vectors must have {expected} dimensions")
        self._chunks.extend(new_chunks)
        self._vectors.extend(new_vectors)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

    def search(self, query_vector: Sequence[float], k: int = 3) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._chunks:
            return []
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query has {len(query_vector)} dimensions but the store holds {self.dimensions}"
            )
        results = [
            SearchResult(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self._chunks, self._vectors, strict=True)
        ]
        results.sort(key=lambda result: (-result.score, result.chunk.id))
        return results[:k]

    def save(self, path: str | Path) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self._chunks],
            "vectors": self._vectors,
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InMemoryVectorStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        store.add((Chunk(**data) for data in payload["chunks"]), payload["vectors"])
        return store
