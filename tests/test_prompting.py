import pytest

from raglab.models import Chunk, SearchResult
from raglab.prompting import EMPTY_CONTEXT, build_prompt, format_context


def _result(source: str, text: str) -> SearchResult:
    chunk = Chunk(
        id=f"{source}#0", text=text, document_id=source, position=0, metadata={"source": source}
    )
    return SearchResult(chunk=chunk, score=1.0)


def test_format_context_numbers_each_source() -> None:
    context = format_context([_result("a.md", "Alpha."), _result("b.md", "Bravo.")])

    assert context == "[1] source: a.md\nAlpha.\n\n[2] source: b.md\nBravo."


def test_build_prompt_includes_context_question_and_rules() -> None:
    prompt = build_prompt("  What is alpha?  ", [_result("a.md", "Alpha is the first letter.")])

    assert "[1] source: a.md\nAlpha is the first letter." in prompt
    assert prompt.rstrip().endswith("Question: What is alpha?\nAnswer:")
    assert "only the context" in prompt


def test_build_prompt_without_results_says_nothing_was_found() -> None:
    assert EMPTY_CONTEXT in build_prompt("Anything?", [])


def test_build_prompt_accepts_custom_templates() -> None:
    prompt = build_prompt("Q?", [_result("a.md", "A.")], template="{question}|{context}")

    assert prompt == "Q?|[1] source: a.md\nA."


def test_build_prompt_rejects_templates_without_placeholders() -> None:
    with pytest.raises(ValueError):
        build_prompt("Q?", [], template="Only {question}")
