from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from raglab.models import Chunk, SearchResult
from raglab.vectorstore import VectorStore

_INTERNAL_KEYS = frozenset({"document_id", "position"})


def _create_chroma_client(path: str | None, host: str | None, port: int) -> Any:
    try:
        chromadb = importlib.import_module("chromadb")
    except ImportError as error:
        raise RuntimeError(
            "Chroma client is not installed. Install it with: pip install 'raglab[chroma]'"
        ) from error
    if host is not None:
        return chromadb.HttpClient(host=host, port=port)
    if path is not None:
        return chromadb.PersistentClient(path=path)
    return chromadb.EphemeralClient()


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        collection_name: str = "raglab_chunks",
        *,
        client: Any = None,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
    ) -> None:
        self.collection_name = collection_name
        self._client = client if client is not None else _create_chroma_client(path, host, port)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dimensions: int | None = None

    def __len__(self) -> int:
        return int(self._collection.count())

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

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
        self._dimensions = expected

        metadatas = [
            {
                **chunk.metadata,
                "document_id": chunk.document_id,
                "position": chunk.position,
            }
            for chunk in chunk_list
        ]
        self._collection.upsert(
            ids=[chunk.id for chunk in chunk_list],
            embeddings=vector_list,
            documents=[chunk.text for chunk in chunk_list],
            metadatas=metadatas,
        )

    def clear(self) -> None:
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._dimensions = None

    @staticmethod
    def _build_where(filters: Mapping[str, str] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = [{key: {"$eq": value}} for key, value in sorted(filters.items())]
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 3,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(self) == 0:
            return []
        kwargs: dict[str, Any] = {
            "query_embeddings": [[float(val) for val in query_vector]],
            "n_results": k,
        }
        where = self._build_where(filters)
        if where is not None:
            kwargs["where"] = where
        response = self._collection.query(**kwargs)
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for chunk_id, text, raw_meta, distance in zip(ids, docs, metas, distances, strict=True):
            meta_dict = dict(raw_meta or {})
            clean_meta = {
                str(key): str(val) for key, val in meta_dict.items() if key not in _INTERNAL_KEYS
            }
            chunk = Chunk(
                id=str(chunk_id),
                text=str(text),
                document_id=str(meta_dict.get("document_id", chunk_id)),
                position=int(meta_dict.get("position", 0)),
                metadata=clean_meta,
            )
            results.append(SearchResult(chunk=chunk, score=1.0 - float(distance)))
        return results
