from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from raglab import (
    Answer,
    BM25Retriever,
    Chunk,
    CrossTermReranker,
    DenseRetriever,
    Document,
    ExtractiveGenerator,
    HashingEmbedder,
    HybridRetriever,
    RAGPipeline,
    Retriever,
    SearchResult,
    TfidfEmbedder,
    evaluate_by_category,
    evaluate_pipeline,
    evaluate_retrieval,
    load_cases,
    load_directory,
)

PROJECT_DIR = Path(__file__).resolve().parent
CORPUS_DIR = PROJECT_DIR / "corpus"
BENCHMARK_PATH = PROJECT_DIR / "benchmark.json"


def load_incident_corpus(root: Path | None = None) -> list[Document]:
    target = root if root is not None else CORPUS_DIR
    return load_directory(target)


PLAYBOOK_QUESTIONS: tuple[tuple[str, dict[str, str] | None], ...] = (
    (
        "Workers emit SQLSTATE 53300 on Postgres connections. "
        "What is the immediate PgBouncer mitigation?",
        {"kind": "runbook", "service": "postgres"},
    ),
    (
        "Database CPU spikes to 95 percent whenever a popular cached merchant "
        "key expires. How do we stop the stampede?",
        {"kind": "runbook"},
    ),
    (
        "Why were mobile shoppers charged twice when retrying timed out card payments in April?",
        {"kind": "postmortem"},
    ),
    (
        "How must new Postgres indexes and foreign key constraints be added "
        "without locking checkout tables?",
        {"kind": "architecture", "service": "postgres"},
    ),
    (
        "What burn rate threshold pages Sev1 immediately on checkout availability?",
        {"kind": "architecture"},
    ),
    (
        "How do we provision NVIDIA H100 GPU nodes for fraud model training?",
        None,
    ),
)


class _UnfilteredAdapter(Retriever):
    def __init__(self, inner: Retriever) -> None:
        self.inner = inner

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.inner.index(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 3,
        *,
        filters: Any = None,
    ) -> list[SearchResult]:
        return self.inner.retrieve(query, k)


def build_incident_copilot(
    *,
    backend: str = "memory",
    qdrant_url: str | None = None,
    max_chars: int = 340,
    overlap: int = 1,
    top_k: int = 3,
) -> RAGPipeline:
    if backend == "qdrant":
        from raglab.integrations.qdrant import QdrantVectorStore

        dense = DenseRetriever(
            HashingEmbedder(),
            QdrantVectorStore(collection_name="atlas_pay_incidents", url=qdrant_url),
        )
        retriever: Retriever = HybridRetriever([dense, BM25Retriever()])
    elif backend == "memory":
        retriever = HybridRetriever()
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'memory' or 'qdrant'")
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=ExtractiveGenerator(
            max_sentences=2,
            min_shared_terms=2,
            min_overlap_ratio=0.34,
        ),
        reranker=CrossTermReranker(),
        candidate_pool=16,
        mmr_lambda=0.65,
        max_context_tokens=260,
        markdown_aware=True,
        max_chars=max_chars,
        overlap=overlap,
        top_k=top_k,
    )
    pipeline.ingest(load_incident_corpus())
    return pipeline


def run_autonomous_playbook(pipeline: RAGPipeline | None = None) -> list[Answer]:
    engine = pipeline if pipeline is not None else build_incident_copilot()
    return [engine.ask(question, filters=filters) for question, filters in PLAYBOOK_QUESTIONS]


def _strip_section_metadata(documents: list[Document]) -> list[Document]:
    return [
        Document(id=doc.id, text=doc.text, metadata={"source": doc.metadata["source"]})
        for doc in documents
    ]


def _build_ablation_pipelines(documents: list[Document]) -> dict[str, RAGPipeline]:
    plain_docs = _strip_section_metadata(documents)
    configs: dict[str, tuple[RAGPipeline, list[Document]]] = {
        "dense": (
            RAGPipeline(
                retriever=_UnfilteredAdapter(
                    DenseRetriever(HashingEmbedder(dimensions=128, use_bigrams=False))
                ),
                generator=ExtractiveGenerator(),
                max_chars=260,
                overlap=1,
                top_k=3,
            ),
            plain_docs,
        ),
        "tfidf": (
            RAGPipeline(
                retriever=_UnfilteredAdapter(DenseRetriever(TfidfEmbedder(use_bigrams=False))),
                generator=ExtractiveGenerator(),
                max_chars=260,
                overlap=1,
                top_k=3,
            ),
            plain_docs,
        ),
        "bm25": (
            RAGPipeline(
                retriever=_UnfilteredAdapter(BM25Retriever()),
                generator=ExtractiveGenerator(),
                max_chars=260,
                overlap=1,
                top_k=3,
            ),
            plain_docs,
        ),
        "hybrid_rrf": (
            RAGPipeline(
                retriever=HybridRetriever(),
                generator=ExtractiveGenerator(),
                max_chars=260,
                overlap=1,
                top_k=3,
            ),
            documents,
        ),
        "senior_full": (
            RAGPipeline(
                retriever=HybridRetriever(),
                generator=ExtractiveGenerator(
                    max_sentences=2,
                    min_shared_terms=2,
                    min_overlap_ratio=0.34,
                ),
                reranker=CrossTermReranker(),
                candidate_pool=16,
                mmr_lambda=0.65,
                max_context_tokens=260,
                markdown_aware=True,
                max_chars=340,
                overlap=1,
                top_k=3,
            ),
            documents,
        ),
    }
    built: dict[str, RAGPipeline] = {}
    for name, (pipeline, corpus) in configs.items():
        pipeline.ingest(corpus)
        built[name] = pipeline
    return built


def run_ablation_suite() -> dict[str, Any]:
    documents = load_incident_corpus()
    cases = load_cases(BENCHMARK_PATH)
    pipelines = _build_ablation_pipelines(documents)

    summary: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        hit_at_1 = evaluate_retrieval(pipeline.retrieve, cases, k=1).hit_rate
        e2e = evaluate_pipeline(pipeline, cases, k=3)
        by_category = evaluate_by_category(pipeline.retrieve, cases, k=3)
        summary[name] = {
            "chunks": len(pipeline.chunks),
            "hit_rate_at_1": round(hit_at_1, 4),
            "hit_rate_at_3": round(e2e.retrieval.hit_rate, 4),
            "mrr_at_3": round(e2e.retrieval.mrr, 4),
            "ndcg_at_3": round(e2e.retrieval.ndcg, 4),
            "recall_at_3": round(e2e.retrieval.recall, 4),
            "precision_at_3": round(e2e.retrieval.precision, 4),
            "grounded_ratio": round(e2e.grounded_ratio, 4),
            "verified_rate": round(e2e.verified_rate, 4),
            "abstention_accuracy": round(e2e.abstention_accuracy, 4),
            "mean_latency_ms": round(e2e.mean_latency_ms, 3),
            "categories_ndcg_at_3": {
                cat: round(report.ndcg, 4) for cat, report in by_category.items()
            },
        }
    return {
        "total_documents": len(documents),
        "total_cases": len(cases),
        "answerable_cases": sum(1 for case in cases if case.answerable),
        "unanswerable_cases": sum(1 for case in cases if not case.answerable),
        "configurations": summary,
    }


def run_chunk_sweep(
    sizes: tuple[int, ...] = (140, 220, 340, 520, 800, 1400),
) -> list[dict[str, float | int]]:
    documents = load_incident_corpus()
    cases = load_cases(BENCHMARK_PATH)
    answerable = [case for case in cases if case.answerable]
    rows: list[dict[str, float | int]] = []
    for size in sizes:
        pipeline = RAGPipeline(
            retriever=HybridRetriever(),
            generator=ExtractiveGenerator(
                max_sentences=2,
                min_shared_terms=2,
                min_overlap_ratio=0.34,
            ),
            reranker=CrossTermReranker(),
            candidate_pool=16,
            mmr_lambda=0.65,
            markdown_aware=False,
            max_chars=size,
            overlap=1,
            top_k=3,
        )
        pipeline.ingest(documents)
        report = evaluate_retrieval(pipeline.retrieve, answerable, k=3)
        hit_1 = evaluate_retrieval(pipeline.retrieve, answerable, k=1).hit_rate
        token_counts = [
            pipeline.ask(case.question, filters=case.filter_map or None).trace.context_tokens
            for case in answerable
        ]
        avg_tokens = sum(token_counts) / len(token_counts)
        rows.append(
            {
                "max_chars": size,
                "chunks": len(pipeline.chunks),
                "hit_rate_at_1": round(hit_1, 4),
                "mrr_at_3": round(report.mrr, 4),
                "ndcg_at_3": round(report.ndcg, 4),
                "recall_at_3": round(report.recall, 4),
                "precision_at_3": round(report.precision, 4),
                "avg_context_tokens": round(avg_tokens, 1),
            }
        )
    return rows
