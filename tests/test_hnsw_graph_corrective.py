from __future__ import annotations

import random
from pathlib import Path

import pytest

from raglab import (
    Chunk,
    CrossTermReranker,
    DenseRetriever,
    Document,
    ExtractiveGenerator,
    HashingEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
)
from raglab.corrective import CorrectiveRAG
from raglab.graph import EntityGraphRetriever, KnowledgeGraph, extract_entities
from raglab.hnsw import HNSWVectorStore
from raglab.query import AcronymQueryExpander
from raglab.vectormath import normalize

CORPUS_DIR = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"


def test_hnsw_vector_store_matches_exact_search_with_high_recall_and_filters() -> None:
    rng = random.Random(17)
    dim = 24
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for idx in range(250):
        vec = normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])
        kind = "runbook" if idx % 3 == 0 else "postmortem"
        chunks.append(
            Chunk(
                id=f"doc#{idx}",
                text=f"chunk {idx}",
                document_id=f"doc_{idx}",
                position=0,
                metadata={"source": f"doc_{idx}.md", "kind": kind},
            )
        )
        vectors.append(vec)

    exact = InMemoryVectorStore()
    hnsw = HNSWVectorStore(m=16, ef_construction=64, ef_search=48, seed=17)
    exact.add(chunks, vectors)
    hnsw.add(chunks, vectors)

    assert len(hnsw) == 250
    assert hnsw.dimensions == dim

    recalls: list[float] = []
    for _ in range(20):
        query = normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])
        exact_ids = {item.chunk.id for item in exact.search(query, k=5)}
        hnsw_ids = {item.chunk.id for item in hnsw.search(query, k=5)}
        recalls.append(len(exact_ids & hnsw_ids) / 5.0)

    assert sum(recalls) / len(recalls) >= 0.85

    probe = vectors[0]
    filtered = hnsw.search(probe, k=3, filters={"kind": "runbook"})
    assert len(filtered) == 3
    assert filtered[0].chunk.id == "doc#0"
    assert all(item.chunk.metadata["kind"] == "runbook" for item in filtered)

    hnsw.clear()
    assert len(hnsw) == 0
    assert hnsw.search(probe, k=3) == []


def test_hnsw_vector_store_validates_arguments() -> None:
    with pytest.raises(ValueError):
        HNSWVectorStore(m=1)
    with pytest.raises(ValueError):
        HNSWVectorStore(ef_construction=0)
    with pytest.raises(ValueError):
        HNSWVectorStore(ef_search=0)


def test_entity_graph_retriever_discovers_multi_hop_related_documents() -> None:
    entities = extract_entities(
        "Workers emit `SQLSTATE 53300` on Postgres and switch PgBouncer to transaction mode."
    )
    assert "sqlstate 53300" in entities
    assert "postgres" in entities
    assert "pgbouncer" in entities

    docs = [
        Document(
            id="hop1.md",
            text="Checkout workers emit `SQLSTATE 53300` when PgBouncer queues overflow.",
            metadata={"source": "hop1.md", "kind": "runbook"},
        ),
        Document(
            id="hop2.md",
            text=(
                "PgBouncer sidecars re resolve Patroni primary endpoints after Frankfurt failover."
            ),
            metadata={"source": "hop2.md", "kind": "postmortem"},
        ),
        Document(
            id="unrelated.md",
            text="Kafka consumer group rebalancing uses max.poll.interval.ms tuning.",
            metadata={"source": "unrelated.md", "kind": "runbook"},
        ),
    ]
    retriever = EntityGraphRetriever(hops=2, decay=0.5)
    pipeline = RAGPipeline(retriever=retriever, top_k=2)
    pipeline.ingest(docs)

    assert isinstance(retriever.graph, KnowledgeGraph)
    assert "pgbouncer" in retriever.graph.neighbors("sqlstate 53300")

    results = pipeline.retrieve("SQLSTATE 53300", k=2)
    sources = [item.chunk.source for item in results]
    assert sources == ["hop1.md", "hop2.md"]

    filtered = pipeline.retrieve("SQLSTATE 53300", k=2, filters={"kind": "postmortem"})
    assert [item.chunk.source for item in filtered] == ["hop2.md"]


def test_corrective_rag_rewrites_acronyms_relaxes_wrong_filters_and_abstains_traps() -> None:
    pipeline = RAGPipeline(
        retriever=DenseRetriever(HashingEmbedder(dimensions=256), HNSWVectorStore()),
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        markdown_aware=True,
        max_chars=340,
    )
    pipeline.ingest_directory(CORPUS_DIR)
    crag = CorrectiveRAG(
        pipeline,
        expander=AcronymQueryExpander({"oom": "OOMKilled exit code 137 memory limits"}),
        min_top_score=0.30,
        allow_filter_relaxation=True,
    )

    direct = crag.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?",
        filters={"kind": "runbook", "service": "postgres"},
    )
    assert direct.decision == "accepted"
    assert direct.answer.grounding.is_verified is True

    relaxed = crag.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?",
        filters={"kind": "postmortem", "service": "postgres"},
    )
    assert relaxed.decision == "relaxed_filters"
    assert relaxed.answer.sources[0].chunk.source == "runbooks/postgres_pool_saturation.md"
    assert len(relaxed.steps) >= 2

    abstained = crag.ask("How do we provision NVIDIA H100 GPU nodes for fraud model training?")
    assert abstained.decision == "abstained"
    assert abstained.answer.grounding.abstained is True

    rewritten_question = "Why do checkout pods hit OOM?"
    rewritten = crag.ask(rewritten_question)
    assert rewritten.decision == "rewritten"
    assert rewritten.answer.question == rewritten_question
    assert rewritten.answer.grounding.is_verified is True
    assert len(rewritten.steps) == 2
    assert rewritten.steps[1].action == "rewrite"


def test_hnsw_and_graph_edge_cases() -> None:
    store = HNSWVectorStore(m=4, ef_construction=16, ef_search=16, seed=7)
    chunk = Chunk(id="c#0", text="hello", document_id="c", position=0, metadata={"kind": "note"})
    with pytest.raises(ValueError):
        store.add([chunk], [[]])
    store.add([chunk], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.search([1.0], k=1)
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)
    zero_hits = store.search([0.0, 0.0], k=1)
    assert len(zero_hits) == 1
    assert zero_hits[0].score == 0.0

    graph = KnowledgeGraph()
    graph.add_chunk("a#0", ["PgBouncer", "Postgres"])
    assert graph.neighbors("PgBouncer") == ("postgres",)

    empty_retriever = EntityGraphRetriever(hops=1, decay=0.5)
    assert empty_retriever.retrieve("Postgres", k=2) == []
    with pytest.raises(ValueError):
        empty_retriever.retrieve("Postgres", k=0)
    with pytest.raises(ValueError):
        EntityGraphRetriever(hops=-1)
    with pytest.raises(ValueError):
        EntityGraphRetriever(decay=1.5)
