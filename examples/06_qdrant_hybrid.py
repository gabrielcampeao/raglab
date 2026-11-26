from pathlib import Path

from raglab import (
    BM25Retriever,
    CrossTermReranker,
    DenseRetriever,
    ExtractiveGenerator,
    HashingEmbedder,
    HybridRetriever,
    RAGPipeline,
)
from raglab.integrations import QdrantVectorStore

CORPUS_DIR = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"


def main() -> None:
    embedder = HashingEmbedder(dimensions=512)
    qdrant_store = QdrantVectorStore(collection_name="incident_runbooks", location=":memory:")
    hybrid = HybridRetriever(
        retrievers=[
            DenseRetriever(embedder=embedder, store=qdrant_store),
            BM25Retriever(),
        ]
    )
    pipeline = RAGPipeline(
        retriever=hybrid,
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        mmr_lambda=0.65,
        max_context_tokens=260,
        markdown_aware=True,
        max_chars=340,
    )

    chunk_count = pipeline.ingest_directory(CORPUS_DIR)
    print(f"Indexed {chunk_count} structural chunks into Qdrant ({len(qdrant_store)} points)")
    print()

    question = (
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?"
    )
    answer = pipeline.ask(question, filters={"kind": "runbook", "service": "postgres"})

    print(f"Question: {question}")
    print(f"Answer:   {answer.text}")
    print(
        f"Verified: {answer.grounding.is_verified} | "
        f"Top source: {answer.sources[0].chunk.source} | "
        f"Latency: {answer.trace.total_ms:.2f}ms"
    )


if __name__ == "__main__":
    main()
