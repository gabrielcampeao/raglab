from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.generation import NO_ANSWER
from raglab.models import SearchResult
from raglab.text import tokenize

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CLAIM_SEGMENT_PATTERN = re.compile(r"[^.!?\n]+[.!?]?(?:\s*\[\d+\])*")
_ABSTENTION_PHRASES = (
    "could not find an answer",
    "do not know",
    "don't know",
    "insufficient context",
    "not contain the answer",
)


@dataclass(frozen=True)
class GroundingReport:
    abstained: bool
    cited_indices: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    grounded_sentences: int
    total_sentences: int
    grounded_ratio: float

    @property
    def is_verified(self) -> bool:
        if self.invalid_citations:
            return False
        return self.abstained or (self.total_sentences > 0 and self.grounded_ratio >= 0.5)


def _unique_ints(values: list[str]) -> tuple[int, ...]:
    seen: list[int] = []
    for raw in values:
        number = int(raw)
        if number not in seen:
            seen.append(number)
    return tuple(seen)


def _source_tokens(result: SearchResult) -> set[str]:
    section = result.chunk.metadata.get("section", "")
    return set(tokenize(result.chunk.text)) | set(tokenize(section))


def verify_answer(
    answer_text: str,
    sources: Sequence[SearchResult],
    *,
    min_sentence_support: float = 0.4,
) -> GroundingReport:
    stripped = answer_text.strip()
    lowered = stripped.lower()
    abstained = (
        not stripped
        or stripped == NO_ANSWER
        or any(phrase in lowered for phrase in _ABSTENTION_PHRASES)
    )
    cited_indices = _unique_ints(_CITATION_PATTERN.findall(stripped))
    invalid_citations = tuple(index for index in cited_indices if index < 1 or index > len(sources))

    if abstained and not invalid_citations:
        return GroundingReport(
            abstained=True,
            cited_indices=cited_indices,
            invalid_citations=(),
            grounded_sentences=0,
            total_sentences=0,
            grounded_ratio=1.0,
        )

    source_term_sets = [_source_tokens(source) for source in sources]
    all_source_terms = set().union(*source_term_sets) if source_term_sets else set()

    segments = [
        match.group(0).strip()
        for match in _CLAIM_SEGMENT_PATTERN.finditer(stripped)
        if match.group(0).strip()
    ]

    grounded_count = 0
    total_count = 0
    for segment in segments:
        segment_citations = _unique_ints(_CITATION_PATTERN.findall(segment))
        clean_text = _CITATION_PATTERN.sub("", segment).strip()
        claim_terms = set(tokenize(clean_text))
        if not claim_terms:
            continue
        total_count += 1
        valid_local = [idx for idx in segment_citations if 1 <= idx <= len(sources)]
        if segment_citations and not valid_local:
            continue
        support_pool = (
            set().union(*(source_term_sets[idx - 1] for idx in valid_local))
            if valid_local
            else all_source_terms
        )
        if not support_pool:
            continue
        support_ratio = len(claim_terms & support_pool) / len(claim_terms)
        if support_ratio >= min_sentence_support:
            grounded_count += 1

    ratio = grounded_count / total_count if total_count else 0.0
    return GroundingReport(
        abstained=abstained,
        cited_indices=cited_indices,
        invalid_citations=invalid_citations,
        grounded_sentences=grounded_count,
        total_sentences=total_count,
        grounded_ratio=ratio,
    )
