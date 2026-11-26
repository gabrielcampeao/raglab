from raglab.generation import NO_ANSWER
from raglab.models import Chunk, SearchResult
from raglab.verification import verify_answer


def _source(name: str, text: str) -> SearchResult:
    chunk = Chunk(
        id=f"{name}#0",
        text=text,
        document_id=name,
        position=0,
        metadata={"source": name},
    )
    return SearchResult(chunk=chunk, score=1.0)


def test_verify_answer_confirms_grounded_cited_sentences() -> None:
    sources = [
        _source("pg.md", "Switch PgBouncer pool mode to transaction during saturation."),
        _source("pg.md", "Terminate idle transactions older than thirty seconds."),
    ]
    answer = (
        "Switch PgBouncer pool mode to transaction during saturation. [1] "
        "Terminate idle transactions older than thirty seconds. [2]"
    )

    report = verify_answer(answer, sources)

    assert report.abstained is False
    assert report.cited_indices == (1, 2)
    assert report.invalid_citations == ()
    assert report.grounded_sentences == 2
    assert report.total_sentences == 2
    assert report.grounded_ratio == 1.0
    assert report.is_verified is True


def test_verify_answer_flags_out_of_range_citations() -> None:
    sources = [_source("pg.md", "Switch PgBouncer pool mode to transaction.")]

    report = verify_answer("Switch PgBouncer pool mode to transaction. [9]", sources)

    assert report.invalid_citations == (9,)
    assert report.is_verified is False


def test_verify_answer_detects_ungrounded_fabrication() -> None:
    sources = [_source("pg.md", "Switch PgBouncer pool mode to transaction.")]

    report = verify_answer("Provision four NVIDIA H100 nodes in Oregon. [1]", sources)

    assert report.grounded_sentences == 0
    assert report.grounded_ratio == 0.0
    assert report.is_verified is False


def test_verify_answer_treats_clean_abstention_as_verified() -> None:
    report = verify_answer(NO_ANSWER, [])

    assert report.abstained is True
    assert report.grounded_ratio == 1.0
    assert report.is_verified is True


def test_verify_answer_does_not_split_on_dotted_identifiers_or_versions() -> None:
    pg_source = _source("pg.md", "Switch PgBouncer pool mode to transaction.")
    tls_source = _source("tls.md", "Enable TLS 1.3 and max.poll.interval.ms on edge.")

    wrong_citation = verify_answer(
        "Enable TLS 1.3 and max.poll.interval.ms on edge. [1]",
        [pg_source, tls_source],
    )
    right_citation = verify_answer(
        "Enable TLS 1.3 and max.poll.interval.ms on edge. [2]",
        [pg_source, tls_source],
    )

    assert wrong_citation.total_sentences == 1
    assert wrong_citation.is_verified is False
    assert right_citation.total_sentences == 1
    assert right_citation.is_verified is True


def test_verify_answer_rejects_fabricated_claim_mixed_with_abstention_phrase() -> None:
    sources = [_source("pg.md", "Switch PgBouncer pool mode to transaction.")]

    report = verify_answer(
        "I do not know the exact latency, so provision four NVIDIA H100 GPUs now. [1]",
        sources,
    )

    assert report.abstained is False
    assert report.is_verified is False
