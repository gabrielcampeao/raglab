from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.pipeline import Answer, RAGPipeline
from raglab.retrievers import MetadataFilter
from raglab.text import tokenize

_ANAPHORA_PATTERN = re.compile(
    r"\b(?:it|its|they|them|their|those|these)\b"
    r"|(?:^|\s)(?:about|for|on|with|escalate|fix|mitigate|approve|handle)\s+that\b"
    r"|^(?:what about|how about|and when|and how|and why|and where)\b",
    re.IGNORECASE,
)
_GENERIC_QUERY_WORDS = frozenset(
    {
        "immediate",
        "mitigation",
        "escalate",
        "fix",
        "handle",
        "happen",
        "cause",
        "issue",
        "problem",
        "step",
        "approve",
    }
)


@dataclass(frozen=True)
class ConversationTurn:
    user_question: str
    standalone_query: str
    answer: Answer


def _anchor_terms(turn: ConversationTurn) -> str:
    if turn.answer.grounding.abstained:
        return ""
    if turn.answer.sources:
        top_chunk = turn.answer.sources[0].chunk
        section = top_chunk.metadata.get("section", "")
        if section:
            primary_heading = section.split(">")[0].strip()
            if primary_heading:
                return primary_heading
    tokens = [
        tok
        for tok in tokenize(turn.user_question)
        if tok not in _GENERIC_QUERY_WORDS and len(tok) > 2
    ]
    return " ".join(dict.fromkeys(tokens[:4]))


def condense_follow_up(question: str, history: Sequence[ConversationTurn]) -> str:
    clean = question.strip()
    if not clean or not history:
        return clean
    if not _ANAPHORA_PATTERN.search(clean):
        return clean
    for turn in reversed(history):
        anchor = _anchor_terms(turn)
        if anchor:
            if anchor.lower() in clean.lower():
                return clean
            return f"{clean} ({anchor})"
    return clean


class ConversationalRAG:
    def __init__(self, pipeline: RAGPipeline, *, max_history: int = 6) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be positive")
        self.pipeline = pipeline
        self.max_history = max_history
        self._turns: list[ConversationTurn] = []

    @property
    def turns(self) -> tuple[ConversationTurn, ...]:
        return tuple(self._turns)

    def reset(self) -> None:
        self._turns.clear()

    def ask(
        self,
        question: str,
        k: int | None = None,
        *,
        filters: MetadataFilter | None = None,
    ) -> ConversationTurn:
        recent = self._turns[-self.max_history :]
        standalone = condense_follow_up(question, recent)
        answer = self.pipeline.ask(
            question,
            k=k,
            filters=filters,
            retrieval_query=standalone if standalone != question else None,
        )
        turn = ConversationTurn(
            user_question=question,
            standalone_query=standalone,
            answer=answer,
        )
        self._turns.append(turn)
        return turn
