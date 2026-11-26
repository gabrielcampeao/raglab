from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raglab.models import SearchResult

if TYPE_CHECKING:
    from raglab.pipeline import RAGPipeline

RetrieveFunction = Callable[..., Sequence[SearchResult]]


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_sources: tuple[str, ...]
    category: str = "general"
    answerable: bool = True
    filters: tuple[tuple[str, str], ...] = ()

    @property
    def filter_map(self) -> dict[str, str]:
        return dict(self.filters)


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    cases: int
    hit_rate: float
    mrr: float
    recall: float
    ndcg: float = 0.0
    precision: float = 0.0


@dataclass(frozen=True)
class EndToEndReport:
    cases: int
    answerable_cases: int
    unanswerable_cases: int
    retrieval: RetrievalReport
    grounded_ratio: float
    verified_rate: float
    abstention_accuracy: float
    mean_latency_ms: float


def _accepts_filters(retrieve: RetrieveFunction) -> bool:
    try:
        parameters = inspect.signature(retrieve).parameters
    except (TypeError, ValueError):
        return False
    if "filters" in parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())


def _ndcg_at_k(sources: list[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for rank, source in enumerate(sources, start=1):
        if source in relevant and source not in seen:
            seen.add(source)
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _score_ranked_sources(
    ranked_per_case: list[tuple[EvalCase, list[str]]], k: int
) -> RetrievalReport:
    hits = 0
    reciprocal_ranks = 0.0
    recall_total = 0.0
    ndcg_total = 0.0
    precision_total = 0.0

    for case, sources in ranked_per_case:
        relevant = set(case.relevant_sources)
        if not relevant:
            raise ValueError(f"Case '{case.question}' has no relevant sources")
        top_sources = sources[:k]
        first_rank = next(
            (rank for rank, source in enumerate(top_sources, start=1) if source in relevant),
            None,
        )
        if first_rank is not None:
            hits += 1
            reciprocal_ranks += 1.0 / first_rank
        recall_total += len(relevant.intersection(top_sources)) / len(relevant)
        ndcg_total += _ndcg_at_k(top_sources, relevant, k)
        precision_total += sum(1 for source in top_sources if source in relevant) / k

    total = len(ranked_per_case)
    return RetrievalReport(
        k=k,
        cases=total,
        hit_rate=hits / total,
        mrr=reciprocal_ranks / total,
        recall=recall_total / total,
        ndcg=ndcg_total / total,
        precision=precision_total / total,
    )


def evaluate_retrieval(
    retrieve: RetrieveFunction, cases: Iterable[EvalCase], k: int = 3
) -> RetrievalReport:
    if k <= 0:
        raise ValueError("k must be positive")
    answerable = [case for case in cases if case.answerable]
    if not answerable:
        raise ValueError("At least one answerable evaluation case is required")

    supports_filters = _accepts_filters(retrieve)
    ranked_per_case: list[tuple[EvalCase, list[str]]] = []
    for case in answerable:
        results = (
            retrieve(case.question, k, filters=case.filter_map)
            if case.filters and supports_filters
            else retrieve(case.question, k)
        )
        sources = [result.chunk.source for result in list(results)[:k]]
        ranked_per_case.append((case, sources))
    return _score_ranked_sources(ranked_per_case, k)


def evaluate_by_category(
    retrieve: RetrieveFunction, cases: Iterable[EvalCase], k: int = 3
) -> dict[str, RetrievalReport]:
    if k <= 0:
        raise ValueError("k must be positive")
    buckets: dict[str, list[EvalCase]] = {}
    for case in cases:
        if not case.answerable:
            continue
        buckets.setdefault(case.category, []).append(case)
    return {
        category: evaluate_retrieval(retrieve, bucket, k) for category, bucket in buckets.items()
    }


def evaluate_pipeline(
    pipeline: RAGPipeline, cases: Iterable[EvalCase], k: int = 3
) -> EndToEndReport:
    if k <= 0:
        raise ValueError("k must be positive")
    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one evaluation case is required")

    ranked_answerable: list[tuple[EvalCase, list[str]]] = []
    grounded_scores: list[float] = []
    abstained_traps = 0
    unanswerable_total = 0
    verified_total = 0
    latency_total = 0.0

    for case in case_list:
        filters = case.filter_map if case.filters else None
        answer = pipeline.ask(case.question, k=k, filters=filters)
        latency_total += answer.trace.total_ms
        if case.answerable:
            sources = [result.chunk.source for result in answer.sources[:k]]
            ranked_answerable.append((case, sources))
            if answer.grounding.abstained:
                grounded_scores.append(0.0)
            else:
                grounded_scores.append(answer.grounding.grounded_ratio)
                if answer.grounding.is_verified:
                    verified_total += 1
        else:
            unanswerable_total += 1
            if answer.grounding.abstained and not answer.grounding.invalid_citations:
                abstained_traps += 1
                verified_total += 1

    if not ranked_answerable:
        raise ValueError("At least one answerable evaluation case is required")

    retrieval_report = _score_ranked_sources(ranked_answerable, k)
    abstention_accuracy = abstained_traps / unanswerable_total if unanswerable_total else 1.0
    return EndToEndReport(
        cases=len(case_list),
        answerable_cases=len(ranked_answerable),
        unanswerable_cases=unanswerable_total,
        retrieval=retrieval_report,
        grounded_ratio=sum(grounded_scores) / len(grounded_scores),
        verified_rate=verified_total / len(case_list),
        abstention_accuracy=abstention_accuracy,
        mean_latency_ms=latency_total / len(case_list),
    )


def _parse_filters(raw: Any, index: int) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ValueError(f"Case {index} 'filters' must be a JSON object")
    return tuple(sorted((str(key), str(value)) for key, value in raw.items()))


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("The evaluation file must contain a JSON list")
    cases: list[EvalCase] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Case {index} needs 'question' and 'relevant_sources'")
        try:
            question = item["question"]
            sources = item["relevant_sources"]
        except KeyError as error:
            raise ValueError(f"Case {index} needs 'question' and 'relevant_sources'") from error
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Case {index} 'question' must be a non empty string")
        if not isinstance(sources, list) or not all(isinstance(src, str) for src in sources):
            raise ValueError(f"Case {index} 'relevant_sources' must be a list of strings")
        category = str(item.get("category", "general"))
        answerable = bool(item.get("answerable", True))
        filters = _parse_filters(item.get("filters"), index)
        cases.append(
            EvalCase(
                question=str(question),
                relevant_sources=tuple(str(source) for source in sources),
                category=category,
                answerable=answerable,
                filters=filters,
            )
        )
    return cases
