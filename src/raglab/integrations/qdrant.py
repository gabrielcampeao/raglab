from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from raglab.models import Chunk, SearchResult
from raglab.vectorstore import VectorStore

_RESERVED_KEYS = frozenset({"_chunk_id", "_text", "_document_id", "_position"})


def _load_qdrant_modules() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qmodels
    except ImportError as error:
        raise RuntimeError(
            "Qdrant client is not installed. Install it with: pip install 'raglab[qdrant]'"
        ) from error
    return QdrantClient, qmodels


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "raglab_chunks",
        *,
        client: Any = None,
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
        location: str | None = None,
    ) -> None:
        qdrant_cls, qmodels = _load_qdrant_modules()
        self.collection_name = collection_name
        self._models = qmodels
        self._dimensions: int | None = None
        if client is not None:
            self._client = client
        elif url is not None:
            self._client = qdrant_cls(url=url, api_key=api_key)
        elif path is not None:
            self._client = qdrant_cls(path=path)
        else:
            self._client = qdrant_cls(location=location or ":memory:")

    def __len__(self) -> int:
        if not self._client.collection_exists(self.collection_name):
            return 0
        return int(self._client.count(collection_name=self.collection_name, exact=True).count)

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def _ensure_collection(self, size: int) -> None:
        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._models.VectorParams(
                size=size,
                distance=self._models.Distance.COSINE,
            ),
        )
        self._dimensions = size

    def add(self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]) -> None:
        chunk_list = list(chunks)
        vector_list = [[float(val) for val in vec] for vec in vectors]
        if len(chunk_list) != len(vector_list):
            raise ValueError("Each chunk needs exactly one vector")
        if not vector_list:
            return
        expected = self._dimensions if self._dimensions is not None else len(vector_list[0])
        if any(len(vec) != expected for vec in vector_list):
            raise ValueError(f"All vectors must have {expected} dimensions")
        self._ensure_collection(expected)
        self._dimensions = expected

        points = []
        for chunk, vec in zip(chunk_list, vector_list, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
            payload = {
                **chunk.metadata,
                "_chunk_id": chunk.id,
                "_text": chunk.text,
                "_document_id": chunk.document_id,
                "_position": chunk.position,
            }
            points.append(self._models.PointStruct(id=point_id, vector=vec, payload=payload))
        self._client.upsert(collection_name=self.collection_name, points=points)

    def clear(self) -> None:
        if self._client.collection_exists(self.collection_name):
            self._client.delete_collection(self.collection_name)
        self._dimensions = None

    def _build_filter(self, filters: Mapping[str, str] | None) -> Any:
        if not filters:
            return None
        conditions = [
            self._models.FieldCondition(
                key=key,
                match=self._models.MatchValue(value=value),
            )
            for key, value in sorted(filters.items())
        ]
        return self._models.Filter(must=conditions)

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 3,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._client.collection_exists(self.collection_name):
            return []
        if self._dimensions is not None and len(query_vector) != self._dimensions:
            raise ValueError(
                f"Query has {len(query_vector)} dimensions but collection holds {self._dimensions}"
            )
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=[float(val) for val in query_vector],
            query_filter=self._build_filter(filters),
            limit=k,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            metadata = {
                str(key): str(val) for key, val in payload.items() if key not in _RESERVED_KEYS
            }
            chunk = Chunk(
                id=str(payload["_chunk_id"]),
                text=str(payload["_text"]),
                document_id=str(payload["_document_id"]),
                position=int(payload["_position"]),
                metadata=metadata,
            )
            results.append(SearchResult(chunk=chunk, score=float(point.score)))
        return results
