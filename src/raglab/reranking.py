from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import pairwise

from raglab.models import SearchResult
from raglab.text import tokenize


def _proximity_score(tokens: list[str], query_set: set[str]) -> float:
    positions = [index for index, token in enumerate(tokens) if token in query_set]
    if len(positions) < 2:
        return 0.0
    best = 0.0
    for left, right in pairwise(positions):
        if tokens[left] != tokens[right]:
            span = right - left + 1
            best = max(best, 2.0 / span)
    return best


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    if not union:
        return 0.0
    return len(first & second) / len(union)


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, results: Sequence[SearchResult], k: int | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError


class CrossTermReranker(Reranker):
    def __init__(
        self,
        *,
        prior_weight: float = 0.25,
        coverage_weight: float = 0.35,
        bigram_weight: float = 0.25,
        section_weight: float = 0.15,
    ) -> None:
        self.prior_weight = prior_weight
        self.coverage_weight = coverage_weight
        self.bigram_weight = bigram_weight
        self.section_weight = section_weight

    def rerank(
        self, query: str, results: Sequence[SearchResult], k: int | None = None
    ) -> list[SearchResult]:
        if k is not None and k <= 0:
            raise ValueError("k must be positive")
        if not results:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            selected = list(results)
            return selected[:k] if k is not None else selected

        query_set = set(query_tokens)
        query_bigrams = set(pairwise(query_tokens))
        max_prior = max(result.score for result in results)
        rescored: list[SearchResult] = []

        for result in results:
            prior = result.score / max_prior if max_prior > 0 else 0.0
            body_tokens = tokenize(result.chunk.text)
            section_tokens = tokenize(result.chunk.metadata.get("section", ""))
            combined = set(body_tokens) | set(section_tokens)
            coverage = len(query_set & combined) / len(query_set)
            candidate_bigrams = set(pairwise(body_tokens)) | set(pairwise(section_tokens))
            bigram_overlap = (
                len(query_bigrams & candidate_bigrams) / len(query_bigrams)
                if query_bigrams
                else coverage
            )
            proximity = _proximity_score(body_tokens, query_set)
            phrase_signal = 0.7 * bigram_overlap + 0.3 * proximity
            section_overlap = (
                len(query_set & set(section_tokens)) / len(query_set) if section_tokens else 0.0
            )
            score = (
                self.prior_weight * prior
                + self.coverage_weight * coverage
                + self.bigram_weight * phrase_signal
                + self.section_weight * section_overlap
            )
            rescored.append(SearchResult(chunk=result.chunk, score=score))

        rescored.sort(key=lambda item: (-item.score, item.chunk.id))
        return rescored[:k] if k is not None else rescored


def diversify_mmr(
    results: Sequence[SearchResult], k: int, lambda_weight: float = 0.75
) -> list[SearchResult]:
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError("lambda_weight must be between 0 and 1")
    if not results:
        return []

    candidates = list(results)
    max_score = max(item.score for item in candidates)
    token_sets = [
        set(tokenize(item.chunk.text)) | set(tokenize(item.chunk.metadata.get("section", "")))
        for item in candidates
    ]
    remaining = list(range(len(candidates)))
    chosen: list[int] = []

    while remaining and len(chosen) < k:
        best_index = remaining[0]
        best_mmr = float("-inf")
        for index in remaining:
            relevance = candidates[index].score / max_score if max_score > 0 else 0.0
            redundancy = (
                max(_jaccard(token_sets[index], token_sets[picked]) for picked in chosen)
                if chosen
                else 0.0
            )
            mmr_value = lambda_weight * relevance - (1.0 - lambda_weight) * redundancy
            if mmr_value > best_mmr:
                best_mmr = mmr_value
                best_index = index
        chosen.append(best_index)
        remaining.remove(best_index)

    return [candidates[index] for index in chosen]
