import pytest

from raglab.models import Chunk, SearchResult
from raglab.reranking import CrossTermReranker, diversify_mmr


def _candidate(chunk_id: str, text: str, score: float, section: str = "") -> SearchResult:
    metadata = {"source": f"{chunk_id}.md"}
    if section:
        metadata["section"] = section
    chunk = Chunk(
        id=chunk_id,
        text=text,
        document_id=chunk_id,
        position=0,
        metadata=metadata,
    )
    return SearchResult(chunk=chunk, score=score)


def test_cross_term_reranker_promotes_exact_phrase_and_section_matches() -> None:
    scattered = _candidate(
        "scattered",
        "Connection metrics are logged daily. Saturation alerts go to Slack. Pool size is 40.",
        score=0.033,
        section="Observability > Metrics",
    )
    exact = _candidate(
        "exact",
        "When connection pool saturation occurs, switch PgBouncer to transaction mode.",
        score=0.031,
        section="Postgres > Connection pool saturation",
    )

    reranked = CrossTermReranker().rerank("connection pool saturation", [scattered, exact], k=2)

    assert [result.chunk.id for result in reranked] == ["exact", "scattered"]
    assert reranked[0].score > reranked[1].score


def test_cross_term_reranker_handles_empty_results_and_unknown_query_terms() -> None:
    reranker = CrossTermReranker()
    candidate = _candidate("only", "Enable TLS 1.3 everywhere.", score=0.5)

    assert reranker.rerank("connection pool", []) == []
    assert reranker.rerank("the and of", [candidate])[0].chunk.id == "only"


def test_diversify_mmr_replaces_near_duplicate_with_complementary_chunk() -> None:
    first = _candidate(
        "pg_a",
        "Switch PgBouncer pool mode to transaction and cap max client connections.",
        score=0.95,
    )
    duplicate = _candidate(
        "pg_b",
        "Switch PgBouncer pool mode to transaction and cap client connections now.",
        score=0.92,
    )
    complementary = _candidate(
        "pg_c",
        "Terminate sessions stuck in idle in transaction older than thirty seconds.",
        score=0.78,
    )

    selected = diversify_mmr([first, duplicate, complementary], k=2, lambda_weight=0.55)

    assert [result.chunk.id for result in selected] == ["pg_a", "pg_c"]


@pytest.mark.parametrize(
    ("k", "lambda_weight"),
    [(0, 0.7), (-1, 0.7), (2, -0.1), (2, 1.1)],
)
def test_diversify_mmr_rejects_invalid_arguments(k: int, lambda_weight: float) -> None:
    candidate = _candidate("a", "Some text.", score=1.0)

    with pytest.raises(ValueError):
        diversify_mmr([candidate], k=k, lambda_weight=lambda_weight)
