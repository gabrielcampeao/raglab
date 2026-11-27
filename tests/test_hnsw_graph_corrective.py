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
    Generator,
    HashingEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
    SearchResult,
    SemanticCache,
)
from raglab.corrective import CorrectiveRAG
from raglab.graph import EntityGraphRetriever, KnowledgeGraph, extract_entities
from raglab.hnsw import HNSWVectorStore
from raglab.query import AcronymQueryExpander, QueryExpander
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


def test_hnsw_preserves_recall_across_sequential_clusters_and_rare_filters() -> None:
    rng = random.Random(42)
    dim = 16
    centroids = [normalize([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(4)]
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for cluster_idx, center in enumerate(centroids):
        for offset in range(45):
            idx = cluster_idx * 45 + offset
            vec = normalize([c + rng.gauss(0.0, 0.12) for c in center])
            rare = "needle" if idx == 179 else "common"
            chunks.append(
                Chunk(
                    id=f"c{cluster_idx}#{offset}",
                    text=f"cluster {cluster_idx} item {offset}",
                    document_id=f"d{cluster_idx}",
                    position=offset,
                    metadata={"cluster": str(cluster_idx), "tag": rare},
                )
            )
            vectors.append(vec)

    exact = InMemoryVectorStore()
    hnsw = HNSWVectorStore(m=16, ef_construction=64, ef_search=48, seed=42)
    exact.add(chunks, vectors)
    hnsw.add(chunks, vectors)

    for cluster_idx, center in enumerate(centroids):
        exact_ids = {item.chunk.id for item in exact.search(center, k=5)}
        hnsw_ids = {item.chunk.id for item in hnsw.search(center, k=5)}
        assert len(exact_ids & hnsw_ids) >= 4, f"cluster {cluster_idx} recall dropped"

    rare_hits = hnsw.search(centroids[0], k=3, filters={"tag": "needle"})
    assert [item.chunk.id for item in rare_hits] == ["c3#44"]


def test_entity_graph_ignores_sql_stop_words_normalizes_and_overwrites_cleanly() -> None:
    retriever = EntityGraphRetriever(hops=2, decay=0.45)
    pipeline = RAGPipeline(retriever=retriever, markdown_aware=True, max_chars=340, top_k=3)
    pipeline.ingest_directory(CORPUS_DIR)

    assert pipeline.retrieve("Why did my printer not work?", k=3) == []
    assert pipeline.retrieve("How do I get a refund on my flight?", k=3) == []

    graph = KnowledgeGraph()
    graph.add_chunk("c1", ["PgBouncer", "Postgres"])
    assert graph.neighbors("  PgBouncer  ") == ("postgres",)
    assert "postgres" in graph.multi_hop_scores(["  PgBouncer  "], hops=1, decay=0.5)

    graph.add_chunk("c1", ["Redis", "Singleflight"])
    assert graph.neighbors("pgbouncer") == ()
    assert graph.neighbors("redis") == ("singleflight",)
    assert "pgbouncer" not in graph.entities

    with pytest.raises(ValueError):
        graph.multi_hop_scores(["redis"], hops=-1)
    with pytest.raises(ValueError):
        graph.multi_hop_scores(["redis"], decay=1.2)


class _BlankVariantExpander(QueryExpander):
    def expand(self, query: str) -> list[str]:
        return [query, "   ", f"{query} OOMKilled exit code 137 memory limits"]


class _UngroundedGenerator(Generator):
    def generate(
        self, question: str, prompt: str, results: tuple[SearchResult, ...] | list[SearchResult]
    ) -> str:
        return "Fabricated answer with invalid citation [99]."


class _StubLLMGenerator(Generator):
    def generate(
        self, question: str, prompt: str, results: tuple[SearchResult, ...] | list[SearchResult]
    ) -> str:
        return f"Memory limits triggered OOMKilled exit code 137 for {question} [1]."


def test_corrective_rag_default_rrf_cache_blank_variants_and_guardrails() -> None:
    default_pipeline = RAGPipeline(markdown_aware=True, max_chars=340)
    default_pipeline.ingest_directory(CORPUS_DIR)
    default_crag = CorrectiveRAG(default_pipeline)
    rrf_outcome = default_crag.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?"
    )
    assert rrf_outcome.decision == "accepted"
    assert rrf_outcome.answer.grounding.is_verified is True

    cached_pipeline = RAGPipeline(
        retriever=DenseRetriever(HashingEmbedder(dimensions=256), HNSWVectorStore()),
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        cache=SemanticCache(similarity_threshold=0.55),
        markdown_aware=True,
        max_chars=340,
    )
    cached_pipeline.ingest_directory(CORPUS_DIR)
    cached_crag = CorrectiveRAG(
        cached_pipeline,
        expander=_BlankVariantExpander(),
        min_top_score=0.30,
        allow_filter_relaxation=False,
    )
    rewritten = cached_crag.ask("Why do checkout pods hit OOM?")
    assert rewritten.decision == "rewritten"
    assert rewritten.answer.grounding.is_verified is True
    assert "Question: Why do checkout pods hit OOM?" in rewritten.answer.prompt
    assert "OOMKilled exit code 137" not in rewritten.answer.prompt.split("Question:")[-1]

    cached_repeat = cached_crag.ask("Why do checkout pods hit OOM?")
    assert cached_repeat.decision == "accepted"
    assert cached_repeat.steps[0].action == "cache"
    assert cached_repeat.answer.trace.cache_hit is True

    no_relax = cached_crag.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?",
        filters={"kind": "postmortem", "service": "postgres"},
    )
    assert no_relax.decision == "abstained"

    ungrounded_pipeline = RAGPipeline(
        generator=_UngroundedGenerator(),
        reranker=CrossTermReranker(),
        markdown_aware=True,
        max_chars=340,
    )
    ungrounded_pipeline.ingest_directory(CORPUS_DIR)
    forced_abstain = CorrectiveRAG(ungrounded_pipeline, min_top_score=0.25).ask(
        "Workers emit SQLSTATE 53300 on Postgres connections."
    )
    assert forced_abstain.decision == "abstained"
    assert forced_abstain.answer.grounding.abstained is True

    llm_pipeline = RAGPipeline(
        retriever=DenseRetriever(HashingEmbedder(dimensions=256), HNSWVectorStore()),
        generator=_StubLLMGenerator(),
        reranker=CrossTermReranker(),
        markdown_aware=True,
        max_chars=340,
    )
    llm_pipeline.ingest_directory(CORPUS_DIR)
    llm_crag = CorrectiveRAG(
        llm_pipeline,
        expander=AcronymQueryExpander({"oom": "OOMKilled exit code 137 memory limits"}),
        min_top_score=0.45,
    )
    llm_outcome = llm_crag.ask("Why do checkout pods hit OOM?")
    assert "Question: Why do checkout pods hit OOM?" in llm_outcome.answer.prompt

    with pytest.raises(ValueError):
        CorrectiveRAG(default_pipeline, min_top_score=-0.1)
    with pytest.raises(ValueError):
        CorrectiveRAG(default_pipeline, min_grounded_ratio=1.5)
