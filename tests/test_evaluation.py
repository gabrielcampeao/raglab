import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from raglab.evaluation import EvalCase, evaluate_retrieval, load_cases
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
