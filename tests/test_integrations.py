from __future__ import annotations

from typing import Any

import pytest

from raglab import (
    Chunk,
    CrossTermReranker,
    DenseRetriever,
    ExtractiveGenerator,
    HashingEmbedder,
    HybridRetriever,
    RAGPipeline,
    SearchResult,
)
from raglab.integrations import (
    ChromaVectorStore,
    CohereCompatibleReranker,
    OllamaEmbedder,
    OpenAIEmbedder,
    PgVectorStore,
    QdrantVectorStore,
)


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="runbooks/pg.md#0",
            text="Switch PgBouncer pool_mode to transaction during SQLSTATE 53300 saturation.",
            document_id="runbooks/pg.md",
            position=0,
            metadata={
                "source": "runbooks/pg.md",
                "kind": "runbook",
                "service": "postgres",
                "section": "Postgres saturation > Immediate mitigation",
            },
        ),
        Chunk(
            id="postmortems/pg_failover.md#0",
            text="Postgres failover left stale DNS connections until PgBouncer sidecars rolled.",
            document_id="postmortems/pg_failover.md",
            position=0,
            metadata={
                "source": "postmortems/pg_failover.md",
                "kind": "postmortem",
                "service": "postgres",
                "section": "EU failover > Corrective actions",
            },
        ),
        Chunk(
            id="runbooks/redis.md#0",
            text="Enable singleflight request coalescing to stop Redis hot key stampedes.",
            document_id="runbooks/redis.md",
            position=0,
            metadata={
                "source": "runbooks/redis.md",
                "kind": "runbook",
                "service": "redis",
                "section": "Redis stampede > Immediate mitigation",
            },
        ),
    ]


def test_qdrant_vector_store_indexes_searches_and_pushes_down_payload_filters() -> None:
    embedder = HashingEmbedder(dimensions=64)
    chunks = _sample_chunks()
    vectors = embedder.embed([chunk.text for chunk in chunks])

    store = QdrantVectorStore(collection_name="sre_chunks", location=":memory:")
    store.add(chunks, vectors)

    assert len(store) == 3
    assert store.dimensions == 64

    unfiltered = store.search(embedder.embed_query("PgBouncer SQLSTATE 53300"), k=2)
    assert unfiltered[0].chunk.id == "runbooks/pg.md#0"

    filtered = store.search(
        embedder.embed_query("PgBouncer connections"),
        k=2,
        filters={"kind": "postmortem", "service": "postgres"},
    )
    assert [item.chunk.id for item in filtered] == ["postmortems/pg_failover.md#0"]
    assert filtered[0].chunk.metadata["section"] == "EU failover > Corrective actions"

    store.clear()
    assert len(store) == 0


def test_qdrant_powers_hybrid_rag_pipeline_end_to_end() -> None:
    embedder = HashingEmbedder(dimensions=128)
    qdrant_store = QdrantVectorStore(collection_name="copilot_hybrid", location=":memory:")
    dense_qdrant = DenseRetriever(embedder=embedder, store=qdrant_store)
    hybrid = HybridRetriever(retrievers=[dense_qdrant, DenseRetriever(embedder)])

    pipeline = RAGPipeline(
        retriever=hybrid,
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        markdown_aware=True,
    )
    pipeline.ingest_directory("projects/incident_copilot/corpus")

    answer = pipeline.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?",
        filters={"kind": "runbook", "service": "postgres"},
    )

    assert not answer.grounding.abstained
    assert answer.grounding.is_verified is True
    assert answer.sources[0].chunk.source == "runbooks/postgres_pool_saturation.md"


class _FakePgCursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows

    def fetchone(self) -> tuple[Any, ...]:
        return (len(self.rows),)

    def __enter__(self) -> _FakePgCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _FakePgConnection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.cursor_instance = _FakePgCursor(rows)
        self.commits = 0

    def cursor(self) -> _FakePgCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commits += 1


def test_pgvector_store_creates_hnsw_index_and_executes_filtered_cosine_query() -> None:
    rows = [
        (
            "runbooks/pg.md#0",
            "Switch PgBouncer pool_mode to transaction.",
            "runbooks/pg.md",
            0,
            {"source": "runbooks/pg.md", "kind": "runbook"},
            0.91,
        )
    ]
    conn = _FakePgConnection(rows)
    store = PgVectorStore(conn, table_name="rag_chunks", dimensions=4)
    store.ensure_schema()
    store.add([_sample_chunks()[0]], [[0.5, 0.5, 0.5, 0.5]])

    results = store.search([0.5, 0.5, 0.5, 0.5], k=1, filters={"kind": "runbook"})

    statements = "\n".join(sql for sql, _ in conn.cursor_instance.executed)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in statements
    assert "USING hnsw (embedding vector_cosine_ops)" in statements
    assert "embedding <=>" in statements
    assert "metadata @>" in statements
    assert results[0].chunk.id == "runbooks/pg.md#0"
    assert results[0].score == pytest.approx(0.91)


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.added: dict[str, Any] = {}
        self.last_query: dict[str, Any] = {}

    def count(self) -> int:
        return len(self.added.get("ids", []))

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.added = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        }

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.last_query = kwargs
        return {
            "ids": [["runbooks/pg.md#0"]],
            "documents": [["Switch PgBouncer pool_mode to transaction."]],
            "metadatas": [
                [
                    {
                        "source": "runbooks/pg.md",
                        "kind": "runbook",
                        "document_id": "runbooks/pg.md",
                        "position": 0,
                    }
                ]
            ],
            "distances": [[0.12]],
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection = _FakeChromaCollection()
        self.deleted: list[str] = []

    def get_or_create_collection(
        self, *, name: str, metadata: dict[str, str]
    ) -> _FakeChromaCollection:
        assert metadata == {"hnsw:space": "cosine"}
        return self.collection

    def delete_collection(self, *, name: str) -> None:
        self.deleted.append(name)


def test_chroma_vector_store_builds_compound_where_filter_and_converts_distance() -> None:
    client = _FakeChromaClient()
    store = ChromaVectorStore(collection_name="sre", client=client)
    store.add([_sample_chunks()[0]], [[1.0, 0.0]])

    results = store.search(
        [1.0, 0.0],
        k=1,
        filters={"kind": "runbook", "service": "postgres"},
    )

    assert client.collection.last_query["where"] == {
        "$and": [{"kind": {"$eq": "runbook"}}, {"service": {"$eq": "postgres"}}]
    }
    assert results[0].score == pytest.approx(0.88)


def test_openai_and_ollama_embedders_and_cohere_reranker_parse_protocol_payloads() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def embed_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        calls.append((url, payload))
        if url.endswith("/embeddings"):
            return {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        return {"embeddings": [[0.6, 0.8]]}

    openai_emb = OpenAIEmbedder(
        model="text-embedding-3-small",
        api_key="sk",
        dimensions=2,
        transport=embed_transport,
    )
    vectors = openai_emb.embed(["first", "second"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0][1]["dimensions"] == 2

    ollama_emb = OllamaEmbedder(model="nomic-embed-text", transport=embed_transport)
    assert ollama_emb.embed_query("hello") == [0.6, 0.8]

    def rerank_transport(
        url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        calls.append((url, payload))
        return {
            "results": [
                {"index": 1, "relevance_score": 0.94},
                {"index": 0, "relevance_score": 0.21},
            ]
        }

    reranker = CohereCompatibleReranker(
        model="rerank-v3.5",
        api_key="cohere_key",
        transport=rerank_transport,
    )
    chunks = _sample_chunks()[:2]
    candidates = [
        SearchResult(chunk=chunks[0], score=0.5),
        SearchResult(chunk=chunks[1], score=0.4),
    ]
    reranked = reranker.rerank("failover DNS", candidates, k=2)

    assert [item.chunk.id for item in reranked] == [
        "postmortems/pg_failover.md#0",
        "runbooks/pg.md#0",
    ]
    assert reranked[0].score == pytest.approx(0.94)
