from __future__ import annotations

import random
from pathlib import Path

from raglab import (
    AcronymQueryExpander,
    Chunk,
    CorrectiveRAG,
    CrossTermReranker,
    DenseRetriever,
    EntityGraphRetriever,
    ExtractiveGenerator,
    HashingEmbedder,
    HNSWVectorStore,
    InMemoryVectorStore,
    RAGPipeline,
)
from raglab.vectormath import normalize

CORPUS = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"


def demo_hnsw_vs_exact() -> None:
    rng = random.Random(42)
    dim = 24
    centroids = [normalize([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(12)]
    chunks = [
        Chunk(
            id=f"vec#{idx}",
            text=f"Chunk {idx}",
            document_id=f"doc_{idx}",
            position=0,
            metadata={"kind": "runbook" if idx % 2 == 0 else "postmortem"},
        )
        for idx in range(400)
    ]
    vectors = [
        normalize([c + rng.gauss(0.0, 0.35) for c in centroids[idx % len(centroids)]])
        for idx in range(len(chunks))
    ]
    exact = InMemoryVectorStore()
    hnsw = HNSWVectorStore(m=16, ef_construction=48, ef_search=40, seed=42)
    exact.add(chunks, vectors)
    hnsw.add(chunks, vectors)

    recalls: list[float] = []
    for q_idx in range(20):
        qvec = normalize([c + rng.gauss(0.0, 0.35) for c in centroids[q_idx % len(centroids)]])
        exact_ids = {item.chunk.id for item in exact.search(qvec, k=5)}
        hnsw_ids = {item.chunk.id for item in hnsw.search(qvec, k=5)}
        recalls.append(len(exact_ids & hnsw_ids) / 5.0)
    filtered = hnsw.search(vectors[0], k=3, filters={"kind": "runbook"})

    avg_recall = sum(recalls) / len(recalls)
    print("1. HNSW approximate nearest neighbor index")
    print(f"   vectors={len(hnsw)} max_level={hnsw.max_level} recall@5={avg_recall:.2%}")
    print(f"   filtered top hit={filtered[0].chunk.id} ({filtered[0].chunk.metadata['kind']})")
    print()


def demo_entity_graph() -> None:
    retriever = EntityGraphRetriever(hops=2, decay=0.45)
    pipeline = RAGPipeline(retriever=retriever, markdown_aware=True, max_chars=340, top_k=3)
    pipeline.ingest_directory(CORPUS)
    hits = pipeline.retrieve("SQLSTATE 53300", k=3)
    neighbors = retriever.graph.neighbors("sqlstate 53300")[:5]

    print("2. Multi hop entity knowledge graph retrieval")
    print(f"   entities indexed={len(retriever.graph.entities)}")
    print(f"   neighbors of 'sqlstate 53300': {', '.join(neighbors)}")
    for idx, hit in enumerate(hits, start=1):
        print(f"   [{idx}] {hit.chunk.source} (score={hit.score:.3f})")
    print()


def demo_corrective_rag() -> None:
    pipeline = RAGPipeline(
        retriever=DenseRetriever(HashingEmbedder(dimensions=256), HNSWVectorStore()),
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        markdown_aware=True,
        max_chars=340,
    )
    pipeline.ingest_directory(CORPUS)
    crag = CorrectiveRAG(
        pipeline,
        expander=AcronymQueryExpander({"oom": "OOMKilled exit code 137 memory limits"}),
        min_top_score=0.30,
        allow_filter_relaxation=True,
    )

    rewrite_case = crag.ask("Why do checkout pods hit OOM?")
    relaxed_case = crag.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?",
        filters={"kind": "postmortem", "service": "postgres"},
    )
    trap_case = crag.ask("How do we provision NVIDIA H100 GPU nodes for fraud model training?")

    print("3. Corrective RAG controller loop")
    for label, outcome in (
        ("acronym rewrite", rewrite_case),
        ("wrong filter recovery", relaxed_case),
        ("out of scope trap", trap_case),
    ):
        actions = " -> ".join(step.action for step in outcome.steps)
        print(
            f"   {label}: decision={outcome.decision} "
            f"verified={outcome.answer.grounding.is_verified} steps=[{actions}]"
        )


def main() -> None:
    demo_hnsw_vs_exact()
    demo_entity_graph()
    demo_corrective_rag()


if __name__ == "__main__":
    main()
