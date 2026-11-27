from __future__ import annotations

from dataclasses import dataclass, replace

from raglab.generation import NO_ANSWER, ExtractiveGenerator
from raglab.parent_child import ParentChildRetriever
from raglab.pipeline import Answer, RAGPipeline
from raglab.prompting import build_prompt
from raglab.query import AcronymQueryExpander, HyDERetriever, MultiQueryRetriever, QueryExpander
from raglab.retrievers import HybridRetriever, MetadataFilter, Retriever
from raglab.verification import verify_answer


@dataclass(frozen=True)
class CorrectiveStep:
    action: str
    query: str
    filters: dict[str, str] | None
    top_score: float
    verified: bool
    abstained: bool


@dataclass(frozen=True)
class CorrectiveOutcome:
    answer: Answer
    decision: str
    steps: tuple[CorrectiveStep, ...]


def _record_step(
    action: str, query: str, filters: MetadataFilter | None, answer: Answer
) -> CorrectiveStep:
    top_score = answer.sources[0].score if answer.sources else 0.0
    return CorrectiveStep(
        action=action,
        query=query,
        filters=dict(filters) if filters else None,
        top_score=top_score,
        verified=answer.grounding.is_verified,
        abstained=answer.grounding.abstained,
    )


def _uses_rrf_scores(retriever: Retriever) -> bool:
    if isinstance(retriever, (HybridRetriever, HyDERetriever)):
        return True
    if isinstance(retriever, MultiQueryRetriever):
        return retriever.fusion == "rrf" or _uses_rrf_scores(retriever.retriever)
    if isinstance(retriever, ParentChildRetriever):
        return _uses_rrf_scores(retriever.retriever)
    return False


def _resolve_min_top_score(pipeline: RAGPipeline, explicit: float | None) -> float:
    if explicit is not None:
        if explicit < 0.0:
            raise ValueError("min_top_score cannot be negative")
        return explicit
    if pipeline.reranker is None and _uses_rrf_scores(pipeline.retriever):
        return 0.01
    return 0.25


def _rank_attempt(answer: Answer) -> tuple[bool, float]:
    usable = answer.grounding.is_verified and not answer.grounding.abstained
    top_score = answer.sources[0].score if answer.sources else 0.0
    return usable, top_score


class CorrectiveRAG:
    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        expander: QueryExpander | None = None,
        min_top_score: float | None = None,
        min_grounded_ratio: float = 0.5,
        allow_filter_relaxation: bool = True,
    ) -> None:
        if not 0.0 <= min_grounded_ratio <= 1.0:
            raise ValueError("min_grounded_ratio must be between 0 and 1")
        self.pipeline = pipeline
        self.expander = expander if expander is not None else AcronymQueryExpander()
        self.min_top_score = _resolve_min_top_score(pipeline, min_top_score)
        self.min_grounded_ratio = min_grounded_ratio
        self.allow_filter_relaxation = allow_filter_relaxation

    def _passes_gate(self, answer: Answer) -> bool:
        if not answer.sources:
            return False
        if answer.grounding.abstained or not answer.grounding.is_verified:
            return False
        if answer.grounding.grounded_ratio < self.min_grounded_ratio:
            return False
        return answer.sources[0].score >= self.min_top_score

    def _abstained_answer(self, last_answer: Answer) -> Answer:
        if last_answer.grounding.abstained and not last_answer.grounding.invalid_citations:
            return last_answer
        return replace(
            last_answer,
            text=NO_ANSWER,
            grounding=verify_answer(NO_ANSWER, ()),
        )

    def _probe(
        self,
        question: str,
        probe_query: str,
        k: int | None,
        filters: MetadataFilter | None,
    ) -> Answer:
        if probe_query == question:
            return self.pipeline.ask(question, k=k, filters=filters, use_cache=False)
        if isinstance(self.pipeline.generator, ExtractiveGenerator):
            raw = self.pipeline.ask(probe_query, k=k, filters=filters, use_cache=False)
            clean_prompt = build_prompt(question, raw.sources, self.pipeline.template)
            return replace(raw, question=question, prompt=clean_prompt)
        return self.pipeline.ask(
            question,
            k=k,
            filters=filters,
            retrieval_query=probe_query,
            use_cache=False,
        )

    def _cache_verified(
        self,
        question: str,
        answer: Answer,
        k: int | None,
        filters: MetadataFilter | None,
    ) -> None:
        if self.pipeline.cache is not None and k is None:
            self.pipeline.cache.store(question, answer, filters=filters)

    def ask(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
    ) -> CorrectiveOutcome:
        if self.pipeline.cache is not None and k is None:
            cached = self.pipeline.cache.lookup(question, filters=filters)
            if cached is not None and self._passes_gate(cached):
                step = _record_step("cache", question, filters, cached)
                return CorrectiveOutcome(answer=cached, decision="accepted", steps=(step,))

        steps: list[CorrectiveStep] = []
        initial = self._probe(question, question, k, filters)
        steps.append(_record_step("initial", question, filters, initial))
        if self._passes_gate(initial):
            self._cache_verified(question, initial, k, filters)
            return CorrectiveOutcome(answer=initial, decision="accepted", steps=tuple(steps))

        best_probe = question
        best_rank = _rank_attempt(initial)
        last_answer = initial
        variants = [
            candidate.strip()
            for candidate in self.expander.expand(question)
            if candidate.strip() and candidate.strip().lower() != question.strip().lower()
        ]
        if variants:
            rewritten_query = variants[0]
            rewritten = self._probe(question, rewritten_query, k, filters)
            steps.append(_record_step("rewrite", rewritten_query, filters, rewritten))
            last_answer = rewritten
            if self._passes_gate(rewritten):
                self._cache_verified(question, rewritten, k, filters)
                return CorrectiveOutcome(answer=rewritten, decision="rewritten", steps=tuple(steps))
            if _rank_attempt(rewritten) >= best_rank:
                best_probe = rewritten_query

        if filters and self.allow_filter_relaxation:
            relaxed = self._probe(question, best_probe, k, None)
            steps.append(_record_step("relax_filters", best_probe, None, relaxed))
            last_answer = relaxed
            if self._passes_gate(relaxed):
                self._cache_verified(question, relaxed, k, filters)
                return CorrectiveOutcome(
                    answer=relaxed, decision="relaxed_filters", steps=tuple(steps)
                )

        return CorrectiveOutcome(
            answer=self._abstained_answer(last_answer),
            decision="abstained",
            steps=tuple(steps),
        )
