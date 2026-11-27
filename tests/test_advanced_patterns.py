from __future__ import annotations

from pathlib import Path

import pytest

from raglab import (
    CrossTermReranker,
    Document,
    EvalCase,
    ExtractiveGenerator,
    HybridRetriever,
    MultiQueryRetriever,
    RAGPipeline,
    evaluate_pipeline,
)
from raglab.conversation import ConversationalRAG, condense_follow_up
from raglab.parent_child import ParentChildRetriever, build_parent_child_chunks
from raglab.prompting import reorder_lost_in_middle
from raglab.query import HyDERetriever, SubQuestionDecomposer, decompose_query

CORPUS_DIR = Path(__file__).resolve().parents[1] / "projects" / "incident_copilot" / "corpus"


def test_parent_child_retriever_aligns_with_pipeline_ingest_and_validates_sizes() -> None:
    with pytest.raises(ValueError):
        ParentChildRetriever(HybridRetriever(), parent_max_chars=100, child_max_chars=200)

    doc = Document(
        id="runbooks/pg.md",
        text=(
            "# Postgres saturation\n"
            "## Immediate mitigation\n"
            "Switch PgBouncer pool_mode to transaction immediately.\n"
            "Cap default_pool_size at 60 per primary core group.\n"
            "Set idle_in_transaction_session_timeout to 30s via ALTER ROLE.\n"
        ),
        metadata={"source": "runbooks/pg.md", "kind": "runbook"},
    )
    parents, children = build_parent_child_chunks(
        [doc], parent_max_chars=500, child_max_chars=90, markdown_aware=True
    )
    assert len(parents) == 1
    assert len(children) >= 2

    retriever = ParentChildRetriever(HybridRetriever(), parent_max_chars=500, child_max_chars=90)
    pipeline = RAGPipeline(retriever=retriever)
    assert pipeline.ingest([doc]) == 1

    results = pipeline.retrieve("ALTER ROLE idle_in_transaction_session_timeout", k=1)
    assert len(results) == 1
    assert "Switch PgBouncer pool_mode to transaction" in results[0].chunk.text
    assert "ALTER ROLE" in results[0].chunk.text


def test_sub_question_decomposer_preserves_both_topics_even_with_pipeline_reranker() -> None:
    compound = (
        "How do we stop Redis hot key stampedes, and how do we isolate degraded webhook endpoints?"
    )
    parts = decompose_query(compound)
    assert parts == [
        "How do we stop Redis hot key stampedes?",
        "how do we isolate degraded webhook endpoints?",
    ]
    assert decompose_query("redis stampede") == ["redis stampede"]

    retriever = MultiQueryRetriever(HybridRetriever(), expander=SubQuestionDecomposer())
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=CrossTermReranker(),
        markdown_aware=True,
        top_k=2,
    )
    pipeline.ingest_directory(CORPUS_DIR)

    results = pipeline.retrieve(compound, k=2)
    sources = {item.chunk.source for item in results}
    assert sources == {"runbooks/redis_cache_stampede.md", "runbooks/payment_webhook_backlog.md"}


def test_interleave_fusion_backfills_multiple_sections_from_single_filtered_source() -> None:
    retriever = MultiQueryRetriever(HybridRetriever(), expander=SubQuestionDecomposer())
    pipeline = RAGPipeline(retriever=retriever, markdown_aware=True, max_chars=260, top_k=2)
    pipeline.ingest_directory(CORPUS_DIR)

    results = pipeline.retrieve(
        "What is the immediate PgBouncer mitigation and when do we escalate Postgres CPU?",
        k=2,
        filters={"kind": "runbook", "service": "postgres"},
    )
    assert len(results) == 2


def test_hyde_retriever_handles_punctuation_only_and_bridges_vocabulary_gap() -> None:
    docs = [
        Document(
            id="stampede.md",
            text="Enable singleflight request coalescing and probabilistic early refresh.",
            metadata={"source": "stampede.md"},
        ),
        Document(
            id="tls.md",
            text="Operational procedure and root cause specification for mTLS leaf certificates.",
            metadata={"source": "tls.md"},
        ),
    ]
    default_hyde = HyDERetriever(HybridRetriever())
    default_hyde.index([])
    pipeline_default = RAGPipeline(retriever=HyDERetriever(HybridRetriever()))
    pipeline_default.ingest(docs)
    assert pipeline_default.retrieve("???", k=1) == []

    def hypothesis(question: str) -> str:
        return f"{question} Use singleflight request coalescing to deduplicate cache misses."

    pipeline = RAGPipeline(
        retriever=HyDERetriever(HybridRetriever(), hypothesis_builder=hypothesis)
    )
    pipeline.ingest(docs)
    assert (
        pipeline.retrieve("prevent thundering herd on expired keys", k=1)[0].chunk.source
        == "stampede.md"
    )


def test_reorder_lost_in_middle_preserves_retrieval_ndcg_in_evaluate_pipeline() -> None:
    pipeline = RAGPipeline(reorder_context=True, markdown_aware=True, top_k=4)
    pipeline.ingest_directory(CORPUS_DIR)
    ranked = pipeline.retrieve("Postgres connection saturation and failover", k=4)
    reordered = reorder_lost_in_middle(ranked)
    assert reordered[0] == ranked[0]
    assert reordered[-1] == ranked[1]

    cases = [
        EvalCase(
            "How do PgBouncer transaction pooling and PgBouncer sidecar DNS "
            "resolution protect Postgres availability?",
            ("runbooks/postgres_pool_saturation.md", "postmortems/eu_region_failover_2026_07.md"),
            category="multi_doc",
        )
    ]
    plain = RAGPipeline(reorder_context=False, markdown_aware=True, top_k=4)
    plain.ingest_directory(CORPUS_DIR)
    assert evaluate_pipeline(pipeline, cases, k=4).retrieval.ndcg == pytest.approx(
        evaluate_pipeline(plain, cases, k=4).retrieval.ndcg
    )


def test_conversational_rag_handles_abstained_turns_topic_switches_and_three_turns() -> None:
    pipeline = RAGPipeline(
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        reranker=CrossTermReranker(),
        markdown_aware=True,
        max_chars=340,
    )
    pipeline.ingest_directory(CORPUS_DIR)
    chat = ConversationalRAG(pipeline)

    trap = chat.ask("How do we provision NVIDIA H100 GPU nodes for fraud model training?")
    assert trap.answer.grounding.abstained is True

    topic_switch = chat.ask("Postgres SQLSTATE 53300 immediate mitigation?")
    assert topic_switch.standalone_query == "Postgres SQLSTATE 53300 immediate mitigation?"
    assert topic_switch.answer.sources[0].chunk.source == "runbooks/postgres_pool_saturation.md"

    follow_up = chat.ask("When do we escalate it to the database reliability lead?")
    assert "postgres" in follow_up.standalone_query.lower()
    assert "85 percent" in follow_up.answer.text
    assert "SQLSTATE 53300" not in follow_up.answer.text

    new_service = chat.ask("Redis hot key stampede mitigation?")
    assert new_service.standalone_query == "Redis hot key stampede mitigation?"
    assert condense_follow_up("What about them?", chat.turns) != "What about them?"
