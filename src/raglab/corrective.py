from __future__ import annotations

from dataclasses import dataclass, replace

from raglab.generation import NO_ANSWER
from raglab.pipeline import Answer, RAGPipeline
from raglab.query import AcronymQueryExpander, QueryExpander
from raglab.retrievers import MetadataFilter
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


class CorrectiveRAG:
    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        expander: QueryExpander | None = None,
        min_top_score: float = 0.25,
        min_grounded_ratio: float = 0.5,
        allow_filter_relaxation: bool = True,
    ) -> None:
        if min_top_score < 0.0:
            raise ValueError("min_top_score cannot be negative")
        if not 0.0 <= min_grounded_ratio <= 1.0:
            raise ValueError("min_grounded_ratio must be between 0 and 1")
        self.pipeline = pipeline
        self.expander = expander if expander is not None else AcronymQueryExpander()
        self.min_top_score = min_top_score
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

    def ask(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
    ) -> CorrectiveOutcome:
        steps: list[CorrectiveStep] = []
        initial = self.pipeline.ask(question, k=k, filters=filters)
        steps.append(_record_step("initial", question, filters, initial))
        if self._passes_gate(initial):
            return CorrectiveOutcome(answer=initial, decision="accepted", steps=tuple(steps))

        best_probe = question
        last_answer = initial
        variants = [
            candidate
            for candidate in self.expander.expand(question)
            if candidate.strip().lower() != question.strip().lower()
        ]
        if variants:
            rewritten_query = variants[0]
            best_probe = rewritten_query
            rewritten_raw = self.pipeline.ask(
                rewritten_query,
                k=k,
                filters=filters,
            )
            rewritten = replace(rewritten_raw, question=question)
            steps.append(_record_step("rewrite", rewritten_query, filters, rewritten))
            last_answer = rewritten
            if self._passes_gate(rewritten):
                return CorrectiveOutcome(answer=rewritten, decision="rewritten", steps=tuple(steps))

        if filters and self.allow_filter_relaxation:
            relaxed_raw = self.pipeline.ask(
                best_probe,
                k=k,
                filters=None,
            )
            relaxed = replace(relaxed_raw, question=question)
            steps.append(_record_step("relax_filters", best_probe, None, relaxed))
            last_answer = relaxed
            if self._passes_gate(relaxed):
                return CorrectiveOutcome(
                    answer=relaxed, decision="relaxed_filters", steps=tuple(steps)
                )

        return CorrectiveOutcome(
            answer=self._abstained_answer(last_answer),
            decision="abstained",
            steps=tuple(steps),
        )
