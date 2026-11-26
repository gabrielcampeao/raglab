from __future__ import annotations

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


def format_context(results: Sequence[SearchResult]) -> str:
    if not results:
        return EMPTY_CONTEXT
    blocks = [
        f"[{number}] source: {result.chunk.source}\n{result.chunk.text}"
        for number, result in enumerate(results, start=1)
    ]
    return "\n\n".join(blocks)


def build_prompt(
    question: str, results: Sequence[SearchResult], template: str = DEFAULT_TEMPLATE
) -> str:
    if "{context}" not in template or "{question}" not in template:
        raise ValueError("The template needs {context} and {question} placeholders")
    return template.format(context=format_context(results), question=question.strip())
