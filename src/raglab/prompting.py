from __future__ import annotations

import math
from collections.abc import Sequence

from raglab.models import SearchResult

DEFAULT_TEMPLATE = (
    "You answer questions using only the context below.\n"
    "If the context does not contain the answer, say that you do not know.\n"
    "Cite the sources you used with their numbers in brackets, for example [1].\n"
    "\n"
    "Context:\n"
    "{context}\n"
    "\n"
    "Question: {question}\n"
    "Answer:"
)

EMPTY_CONTEXT = "No relevant context was found."


def estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return math.ceil(len(words) * 1.3)


def _format_block(number: int, result: SearchResult) -> str:
    section = result.chunk.metadata.get("section", "")
    header = (
        f"[{number}] source: {result.chunk.source} ({section})"
        if section
        else f"[{number}] source: {result.chunk.source}"
    )
    return f"{header}\n{result.chunk.text}"


def format_context(results: Sequence[SearchResult]) -> str:
    if not results:
        return EMPTY_CONTEXT
    blocks = [_format_block(number, result) for number, result in enumerate(results, start=1)]
    return "\n\n".join(blocks)


def pack_context(results: Sequence[SearchResult], max_tokens: int) -> list[SearchResult]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not results:
        return []
    packed: list[SearchResult] = []
    for result in results:
        candidate = [*packed, result]
        if packed and estimate_tokens(format_context(candidate)) > max_tokens:
            break
        packed.append(result)
    return packed


def build_prompt(
    question: str, results: Sequence[SearchResult], template: str = DEFAULT_TEMPLATE
) -> str:
    if "{context}" not in template or "{question}" not in template:
        raise ValueError("The template needs {context} and {question} placeholders")
    return template.format(context=format_context(results), question=question.strip())
