from pathlib import Path

from raglab import (
    ConversationalRAG,
    CrossTermReranker,
    ExtractiveGenerator,
    HybridRetriever,
    MultiQueryRetriever,
    ParentChildRetriever,
    RAGPipeline,
    SubQuestionDecomposer,
    load_directory,
)

CORPUS_DIR = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"


def main() -> None:
    documents = load_directory(CORPUS_DIR)

    parent_child = ParentChildRetriever(
        HybridRetriever(),
        parent_max_chars=700,
        child_max_chars=160,
        markdown_aware=True,
    )
    parent_count, child_count = parent_child.ingest_documents(documents)
    print(
        f"1. Small to big hierarchy: {child_count} child chunks -> {parent_count} parent sections"
    )
    hit = parent_child.retrieve("ALTER ROLE atlas_app idle_in_transaction_session_timeout", k=1)[0]
    print(f"   Matched parent: {hit.chunk.source} ({len(hit.chunk.text)} chars preserved)")
    print()

    decomposed_retriever = MultiQueryRetriever(
        HybridRetriever(),
        expander=SubQuestionDecomposer(),
    )
    pipeline = RAGPipeline(
        retriever=decomposed_retriever,
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.30),
        reranker=CrossTermReranker(),
        reorder_context=True,
        markdown_aware=True,
        max_chars=340,
        top_k=4,
    )
    pipeline.ingest(documents)

    compound = (
        "How do we stop Redis hot key stampedes and how do we isolate degraded webhook endpoints?"
    )
    compound_results = pipeline.retrieve(compound, k=2)
    sources = {item.chunk.source for item in compound_results}
    assert sources == {"runbooks/redis_cache_stampede.md", "runbooks/payment_webhook_backlog.md"}
    print("2. Sub question decomposition + round robin interleaving:")
    for item in compound_results:
        print(f"   Retrieved: {item.chunk.source}")
    print()

    chat = ConversationalRAG(pipeline)
    first = chat.ask(
        "Workers emit SQLSTATE 53300 on Postgres connections. What is the immediate mitigation?"
    )
    follow_up = chat.ask("When do we escalate it to the database reliability lead?")
    print("3. Multi turn conversational coreference condensation:")
    print(f"   Turn 1 source: {first.answer.sources[0].chunk.source}")
    print(f"   Turn 2 rewritten query: {follow_up.standalone_query}")
    print(f"   Turn 2 answer: {follow_up.answer.text}")


if __name__ == "__main__":
    main()
