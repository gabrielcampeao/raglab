from collections.abc import Iterable

import pytest

from raglab.embeddings import TfidfEmbedder
from raglab.models import Chunk, Document, SearchResult
from raglab.retrievers import (
    RETRIEVER_NAMES,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
    create_retriever,
)


class StaticRetriever(Retriever):
    def __init__(self, ranked_ids: list[str]) -> None:
        self.ranked_ids = ranked_ids
        self.indexed: list[Chunk] = []

    def index(self, chunks: Iterable[Chunk]) -> None:
        self.indexed = list(chunks)

    def retrieve(
        self, query: str, k: int = 3, *, filters: dict[str, str] | None = None
    ) -> list[SearchResult]:
        by_id = {chunk.id: chunk for chunk in self.indexed}
        return [SearchResult(by_id[chunk_id], 1.0) for chunk_id in self.ranked_ids][:k]


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_ranks_the_relevant_chunk_first(
    name: str, sample_chunks: list[Chunk]
) -> None:
    retriever = create_retriever(name)
    retriever.index(sample_chunks)

    results = retriever.retrieve("How do I get a refund?", k=2)

    assert results
    assert results[0].chunk.source == "refunds.md"


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_rejects_non_positive_k(name: str, sample_chunks: list[Chunk]) -> None:
    retriever = create_retriever(name)
    retriever.index(sample_chunks)

    with pytest.raises(ValueError):
        retriever.retrieve("refund", k=0)


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_handles_an_empty_index(name: str) -> None:
    retriever = create_retriever(name)
    retriever.index([])

    assert retriever.retrieve("refund") == []


def test_create_retriever_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown retriever"):
        create_retriever("magic")


def test_bm25_returns_nothing_for_unknown_terms(sample_chunks: list[Chunk]) -> None:
    retriever = BM25Retriever()
    retriever.index(sample_chunks)

    assert retriever.retrieve("banana smoothie") == []


def test_bm25_gives_rare_terms_more_weight() -> None:
    chunks = [
        Chunk(id="a#0", text="plan plan encryption", document_id="a", position=0),
        Chunk(id="b#0", text="plan plan plan", document_id="b", position=0),
        Chunk(id="c#0", text="plan pricing", document_id="c", position=0),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.retrieve("plan encryption")

    assert results[0].chunk.id == "a#0"


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.5), (1.2, 1.5), (1.2, -0.1)])
def test_bm25_rejects_invalid_parameters(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Retriever(k1=k1, b=b)


def test_dense_retriever_reindexing_replaces_previous_chunks(
    sample_documents: list[Document], sample_chunks: list[Chunk]
) -> None:
    retriever = DenseRetriever(TfidfEmbedder())
    retriever.index(sample_chunks)
    retriever.index(sample_chunks[:1])

    assert len(retriever.store) == 1


def test_hybrid_retriever_fuses_rankings_with_reciprocal_rank(
    sample_chunks: list[Chunk],
) -> None:
    ids = [chunk.id for chunk in sample_chunks]
    first = StaticRetriever([ids[0], ids[1], ids[2]])
    second = StaticRetriever([ids[1], ids[3], ids[0]])
    hybrid = HybridRetriever([first, second], rrf_k=1)
    hybrid.index(sample_chunks)

    results = hybrid.retrieve("anything", k=4)

    assert [result.chunk.id for result in results] == [ids[1], ids[0], ids[3], ids[2]]
    assert results[0].score == pytest.approx(1 / 3 + 1 / 2)


def test_hybrid_retriever_indexes_every_inner_retriever(sample_chunks: list[Chunk]) -> None:
    inner = [StaticRetriever([]), StaticRetriever([])]
    HybridRetriever(inner).index(sample_chunks)

    assert all(retriever.indexed == sample_chunks for retriever in inner)


@pytest.mark.parametrize("arguments", [{"rrf_k": 0}, {"candidate_pool": 0}])
def test_hybrid_retriever_rejects_invalid_parameters(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(**arguments)


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_prefilters_by_metadata(name: str) -> None:
    chunks = [
        Chunk(
            id="pg_runbook#0",
            text="Connection pool saturation triggers SQLSTATE 53300.",
            document_id="pg_runbook",
            position=0,
            metadata={"source": "pg_runbook.md", "kind": "runbook", "service": "postgres"},
        ),
        Chunk(
            id="pg_postmortem#0",
            text="Connection pool saturation caused a 12 minute checkout outage.",
            document_id="pg_postmortem",
            position=0,
            metadata={"source": "pg_postmortem.md", "kind": "postmortem", "service": "postgres"},
        ),
        Chunk(
            id="redis_runbook#0",
            text="Cache stampede saturates upstream connection pools.",
            document_id="redis_runbook",
            position=0,
            metadata={"source": "redis_runbook.md", "kind": "runbook", "service": "redis"},
        ),
    ]
    retriever = create_retriever(name)
    retriever.index(chunks)

    results = retriever.retrieve(
        "connection pool saturation",
        k=2,
        filters={"kind": "postmortem", "service": "postgres"},
    )

    assert [result.chunk.id for result in results] == ["pg_postmortem#0"]


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_returns_empty_when_filters_match_nothing(
    name: str, sample_chunks: list[Chunk]
) -> None:
    retriever = create_retriever(name)
    retriever.index(sample_chunks)

    assert retriever.retrieve("refund", k=2, filters={"kind": "nonexistent"}) == []


@pytest.mark.parametrize("name", RETRIEVER_NAMES)
def test_every_retriever_indexes_section_heading_metadata(name: str) -> None:
    chunks = [
        Chunk(
            id="pg#0",
            text="Switch pool_mode to transaction immediately.",
            document_id="pg",
            position=0,
            metadata={"source": "pg.md", "section": "PgBouncer saturation > Emergency mitigation"},
        ),
        Chunk(
            id="kafka#0",
            text="Increase max.poll.interval.ms during batch backfills.",
            document_id="kafka",
            position=0,
            metadata={"source": "kafka.md", "section": "Consumer lag > Rebalance tuning"},
        ),
    ]
    retriever = create_retriever(name)
    retriever.index(chunks)

    results = retriever.retrieve("PgBouncer emergency mitigation", k=1)

    assert len(results) == 1
    assert results[0].chunk.id == "pg#0"
