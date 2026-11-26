from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.generation import NO_ANSWER
from raglab.models import SearchResult
from raglab.text import tokenize

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_SEGMENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?!\[)|(?<=\])\s+(?!\[)|\n+")
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


def _split_claim_segments(text: str) -> list[str]:
    segments: list[str] = []
    prose_lines: list[str] = []
    fence_lines: list[str] = []
    in_fence = False

    def flush_prose() -> None:
        joined = "\n".join(prose_lines).strip()
        prose_lines.clear()
        if not joined:
            return
        for part in _SEGMENT_SPLIT.split(joined):
            cleaned = part.strip()
            if cleaned:
                segments.append(cleaned)

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if in_fence:
                fence_lines.append(line)
                segments.append("\n".join(fence_lines).strip())
                fence_lines.clear()
                in_fence = False
            else:
                flush_prose()
                fence_lines.append(line)
                in_fence = True
        elif in_fence:
            fence_lines.append(line)
        else:
            prose_lines.append(line)

    if fence_lines:
        segments.append("\n".join(fence_lines).strip())
    flush_prose()
    return segments


def verify_answer(
    answer_text: str,
    sources: Sequence[SearchResult],
    *,
    min_sentence_support: float = 0.4,
) -> GroundingReport:
    stripped = answer_text.strip()
    lowered = stripped.lower()
    cited_indices = _unique_ints(_CITATION_PATTERN.findall(stripped))
    invalid_citations = tuple(index for index in cited_indices if index < 1 or index > len(sources))
    segments = _split_claim_segments(stripped) if stripped else []

    abstained = (
        not stripped
        or stripped == NO_ANSWER
        or (
            not cited_indices
            and len(segments) <= 1
            and any(phrase in lowered for phrase in _ABSTENTION_PHRASES)
        )
    )

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
        abstained=False,
        cited_indices=cited_indices,
        invalid_citations=invalid_citations,
        grounded_sentences=grounded_count,
        total_sentences=total_count,
        grounded_ratio=ratio,
    )
