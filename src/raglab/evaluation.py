from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from raglab.models import SearchResult

RetrieveFunction = Callable[[str, int], Sequence[SearchResult]]


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_sources: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    cases: int
    hit_rate: float
    mrr: float
    recall: float


def evaluate_retrieval(
    retrieve: RetrieveFunction, cases: Iterable[EvalCase], k: int = 3
) -> RetrievalReport:
    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one evaluation case is required")
    if k <= 0:
        raise ValueError("k must be positive")

    hits = 0
    reciprocal_ranks = 0.0
    recall_total = 0.0
    for case in case_list:
        relevant = set(case.relevant_sources)
        if not relevant:
            raise ValueError(f"Case '{case.question}' has no relevant sources")
        sources = [result.chunk.source for result in list(retrieve(case.question, k))[:k]]
        first_rank = next(
            (rank for rank, source in enumerate(sources, start=1) if source in relevant), None
        )
        if first_rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / first_rank
        recall_total += len(relevant.intersection(sources)) / len(relevant)

    total = len(case_list)
    return RetrievalReport(
        k=k,
        cases=total,
        hit_rate=hits / total,
        mrr=reciprocal_ranks / total,
        recall=recall_total / total,
    )


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("The evaluation file must contain a JSON list")
    cases: list[EvalCase] = []
    for index, item in enumerate(raw):
        try:
            question = item["question"]
            sources = item["relevant_sources"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Case {index} needs 'question' and 'relevant_sources'") from error
        cases.append(
            EvalCase(question=str(question), relevant_sources=tuple(str(s) for s in sources))
        )
    return cases
