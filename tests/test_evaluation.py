import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from raglab.evaluation import (
    EvalCase,
    evaluate_by_category,
    evaluate_pipeline,
    evaluate_retrieval,
    load_cases,
)
from raglab.models import Chunk, SearchResult


def _result(source: str) -> SearchResult:
    chunk = Chunk(
        id=f"{source}#0", text="", document_id=source, position=0, metadata={"source": source}
    )
    return SearchResult(chunk=chunk, score=1.0)


def _fake_retrieve(ranked: dict[str, list[str]]) -> Callable[[str, int], Sequence[SearchResult]]:
    def retrieve(question: str, k: int) -> list[SearchResult]:
        return [_result(source) for source in ranked[question]]

    return retrieve


def test_evaluate_retrieval_computes_hit_rate_mrr_and_recall() -> None:
    cases = [
        EvalCase("q1", ("a.md",)),
        EvalCase("q2", ("b.md", "c.md")),
        EvalCase("q3", ("d.md",)),
    ]
    retrieve = _fake_retrieve(
        {"q1": ["a.md", "x.md"], "q2": ["x.md", "c.md"], "q3": ["x.md", "y.md"]}
    )

    report = evaluate_retrieval(retrieve, cases, k=2)

    assert report.cases == 3
    assert report.k == 2
    assert report.hit_rate == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.recall == pytest.approx((1 + 0.5 + 0) / 3)


def test_evaluate_retrieval_only_counts_the_top_k_results() -> None:
    retrieve = _fake_retrieve({"q": ["x.md", "a.md"]})

    report = evaluate_retrieval(retrieve, [EvalCase("q", ("a.md",))], k=1)

    assert report.hit_rate == 0.0


def test_evaluate_retrieval_rejects_empty_case_list() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval(_fake_retrieve({}), [])


def test_evaluate_retrieval_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval(_fake_retrieve({"q": []}), [EvalCase("q", ("a.md",))], k=0)


def test_evaluate_retrieval_rejects_cases_without_sources() -> None:
    with pytest.raises(ValueError):
        evaluate_retrieval(_fake_retrieve({"q": []}), [EvalCase("q", ())])


def test_load_cases_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"question": "How?", "relevant_sources": ["a.md", "b.md"]}]),
        encoding="utf-8",
    )

    assert load_cases(path) == [EvalCase("How?", ("a.md", "b.md"))]


@pytest.mark.parametrize("content", [{"question": "x"}, [{"question": "x"}], ["text"]])
def test_load_cases_rejects_invalid_files(tmp_path: Path, content: object) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cases(path)


def test_evaluate_retrieval_computes_ndcg_precision_and_skips_unanswerable_cases() -> None:
    cases = [
        EvalCase("q1", ("a.md",), category="exact"),
        EvalCase("q2", ("b.md",), category="paraphrase"),
        EvalCase("q_trap", (), category="unanswerable", answerable=False),
    ]
    retrieve = _fake_retrieve(
        {"q1": ["a.md", "x.md"], "q2": ["x.md", "b.md"], "q_trap": ["x.md", "y.md"]}
    )

    report = evaluate_retrieval(retrieve, cases, k=2)

    assert report.cases == 2
    assert report.hit_rate == pytest.approx(1.0)
    assert report.precision == pytest.approx(0.5)
    assert 0.8 < report.ndcg < 0.9


def test_evaluate_by_category_groups_answerable_slices_and_forwards_filters() -> None:
    seen_filters: list[dict[str, str] | None] = []

    def filtered_retrieve(
        question: str, k: int, *, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        seen_filters.append(filters)
        return [_result("pg.md")]

    cases = [
        EvalCase("q1", ("pg.md",), category="scoped", filters=(("kind", "runbook"),)),
        EvalCase("q2", (), category="unanswerable", answerable=False),
    ]

    grouped = evaluate_by_category(filtered_retrieve, cases, k=2)

    assert set(grouped) == {"scoped"}
    assert grouped["scoped"].hit_rate == 1.0
    assert seen_filters == [{"kind": "runbook"}]


def test_evaluate_pipeline_scores_grounding_and_abstention() -> None:
    from raglab.generation import ExtractiveGenerator
    from raglab.models import Document
    from raglab.pipeline import RAGPipeline

    pipeline = RAGPipeline(
        generator=ExtractiveGenerator(min_shared_terms=2, min_overlap_ratio=0.34),
        markdown_aware=True,
    )
    pipeline.ingest(
        [
            Document(
                id="pg.md",
                text="# Pool saturation\nSwitch PgBouncer pool mode to transaction.",
                metadata={"source": "pg.md", "kind": "runbook"},
            )
        ]
    )
    cases = [
        EvalCase(
            "How do we mitigate PgBouncer pool mode saturation?",
            ("pg.md",),
            category="exact",
        ),
        EvalCase(
            "Which vendor handles biometric hardware tokens?",
            (),
            category="unanswerable",
            answerable=False,
        ),
    ]

    report = evaluate_pipeline(pipeline, cases, k=2)

    assert report.cases == 2
    assert report.answerable_cases == 1
    assert report.unanswerable_cases == 1
    assert report.retrieval.hit_rate == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.verified_rate == 1.0
