from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from raglab.models import Chunk, SearchResult
from raglab.vectorstore import VectorStore

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(val):.8f}" for val in values) + "]"


class PgVectorStore(VectorStore):
    def __init__(
        self,
        connection: Any,
        *,
        table_name: str = "raglab_chunks",
        dimensions: int = 1024,
    ) -> None:
        if not _IDENTIFIER_PATTERN.match(table_name):
            raise ValueError(f"Unsafe SQL identifier: {table_name}")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._connection = connection
        self.table_name = table_name
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def ensure_schema(self, *, hnsw_m: int = 16, ef_construction: int = 64) -> None:
        table = self.table_name
        with self._connection.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "id TEXT PRIMARY KEY, "
                "text TEXT NOT NULL, "
                "document_id TEXT NOT NULL, "
                "position INTEGER NOT NULL, "
                "metadata JSONB NOT NULL, "
                f"embedding vector({self._dimensions}) NOT NULL)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx "
                f"ON {table} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {int(hnsw_m)}, ef_construction = {int(ef_construction)})"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx "
                f"ON {table} USING gin (metadata jsonb_path_ops)"
            )
        self._connection.commit()

    def __len__(self) -> int:
        with self._connection.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table_name}")
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def add(self, chunks: Iterable[Chunk], vectors: Iterable[Sequence[float]]) -> None:
        chunk_list = list(chunks)
        vector_list = list(vectors)
        if len(chunk_list) != len(vector_list):
            raise ValueError("Each chunk needs exactly one vector")
        if not vector_list:
            return
        if any(len(vec) != self._dimensions for vec in vector_list):
            raise ValueError(f"All vectors must have {self._dimensions} dimensions")
        sql = (
            f"INSERT INTO {self.table_name} "
            "(id, text, document_id, position, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector) "
            "ON CONFLICT (id) DO UPDATE SET "
            "text = EXCLUDED.text, "
            "metadata = EXCLUDED.metadata, "
            "embedding = EXCLUDED.embedding"
        )
        with self._connection.cursor() as cur:
            for chunk, vec in zip(chunk_list, vector_list, strict=True):
                cur.execute(
                    sql,
                    (
                        chunk.id,
                        chunk.text,
                        chunk.document_id,
                        chunk.position,
                        json.dumps(chunk.metadata),
                        _vector_literal(vec),
                    ),
                )
        self._connection.commit()

    def clear(self) -> None:
        with self._connection.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table_name}")
        self._connection.commit()

    def search(
        self,
        query_vector: Sequence[float],
        k: int = 3,
        *,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(query_vector) != self._dimensions:
            raise ValueError(
                f"Query has {len(query_vector)} dimensions but table holds {self._dimensions}"
            )
        literal = _vector_literal(query_vector)
        if filters:
            sql = (
                f"SELECT id, text, document_id, position, metadata, "
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table_name} "
                "WHERE metadata @> %s::jsonb "
                "ORDER BY embedding <=> %s::vector LIMIT %s"
            )
            params: tuple[Any, ...] = (literal, json.dumps(dict(filters)), literal, k)
        else:
            sql = (
                f"SELECT id, text, document_id, position, metadata, "
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {self.table_name} "
                "ORDER BY embedding <=> %s::vector LIMIT %s"
            )
            params = (literal, literal, k)

        with self._connection.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results: list[SearchResult] = []
        for chunk_id, text, document_id, position, raw_meta, score in rows:
            parsed_meta = json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta)
            chunk = Chunk(
                id=str(chunk_id),
                text=str(text),
                document_id=str(document_id),
                position=int(position),
                metadata={str(key): str(val) for key, val in parsed_meta.items()},
            )
            results.append(SearchResult(chunk=chunk, score=float(score)))
        return results
